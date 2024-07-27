package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"cloud.google.com/go/translate"
	"github.com/gin-gonic/gin"
	"github.com/go-resty/resty/v2"
	"github.com/joho/godotenv"
	"golang.org/x/text/language"
	"google.golang.org/api/option"
)

type Detection struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
	Box        [4]int  `json:"box"`
}

type PythonResponse struct {
	Detections     []Detection `json:"detections"`
	AnnotatedImage string      `json:"annotated_image"`
}

type DetectionRequest struct {
	Image string `json:"image"`
}

type TranslateRequest struct {
	Detections []string `json:"detections"`
	Lang       string   `json:"lang"`
}

var translateClient *translate.Client

func setupGoogleTranslate() {
	if err := godotenv.Load(); err != nil {
		fmt.Printf("Error loading .env file: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	var err error

	credsJSON := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
	var credsMap map[string]interface{}

	if err := json.Unmarshal([]byte(credsJSON), &credsMap); err != nil {
		fmt.Printf("Error parsing service account JSON: %v\n", err)
		os.Exit(1)
	}
	credsBytes, err := json.Marshal(credsMap)

	if err != nil {
		fmt.Printf("Error converting service account to bytes: %v\n", err)
		os.Exit(1)
	}

	translateClient, err = translate.NewClient(ctx, option.WithCredentialsJSON(credsBytes))
	if err != nil {
		fmt.Printf("Failed to create client: %v\n", err)
		os.Exit(1)
	}
}

func translateText(detections []string, lang string) ([]string, error) {
	ctx := context.Background()
	translations := make([]string, len(detections))
	langTag := language.Make(lang)
	for i, detection := range detections {
		resp, err := translateClient.Translate(ctx, []string{detection}, langTag, nil)
		if err != nil {
			return nil, err
		}
		if len(resp) > 0 {
			translations[i] = resp[0].Text
		} else {
			translations[i] = detection
		}
	}
	return translations, nil
}

func detectObjects(imageData string) ([]PythonResponse, error) {
	client := resty.New()
	resp, err := client.R().
		SetHeader("Content-Type", "application/json").
		SetBody(DetectionRequest{Image: imageData}).
		Post("http://localhost:5000/detect")

	if err != nil {
		return nil, err
	}

	var pyResp PythonResponse
	if err := json.Unmarshal(resp.Body(), &pyResp); err != nil {
		return nil, err
	}

	return []PythonResponse{pyResp}, nil
}

func main() {
	setupGoogleTranslate()
	r := gin.Default()

	r.Static("/static", "./static")

	r.GET("/", func(c *gin.Context) {
		c.File("./static/index.html")
	})

	r.POST("/upload", func(c *gin.Context) {
		file, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Image upload failed"})
			return
		}

		fileData, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open image file"})
			return
		}
		defer fileData.Close()

		buffer := bytes.NewBuffer(nil)
		if _, err := io.Copy(buffer, fileData); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read image file"})
			return
		}

		encodedImage := base64.StdEncoding.EncodeToString(buffer.Bytes())
		detections, err := detectObjects(encodedImage)
		if err != nil {
			fmt.Println(err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Object detection failed"})
			return
		}
		c.JSON(http.StatusOK, detections)
	})

	r.POST("/translate", func(c *gin.Context) {
		var translateReq TranslateRequest
		if err := c.ShouldBindJSON(&translateReq); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
			return
		}

		translations, err := translateText(translateReq.Detections, translateReq.Lang)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Translation failed"})
			return
		}

		c.JSON(http.StatusOK, gin.H{"translations": translations})
	})

	r.Run(":8080")
}
