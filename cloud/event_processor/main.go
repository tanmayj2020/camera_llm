// Package main implements the cloud event processor.
// Consumes events from Pub/Sub, stores keyframes in GCS, metadata in BigQuery,
// and fans out to downstream consumers.
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"cloud.google.com/go/bigquery"
	"cloud.google.com/go/pubsub"
	"cloud.google.com/go/storage"
)

type VisionEvent struct {
	EventID       string          `json:"event_id" bigquery:"event_id"`
	Timestamp     float64         `json:"timestamp"`
	CameraID      string          `json:"camera_id" bigquery:"camera_id"`
	EventType     string          `json:"event_type" bigquery:"event_type"`
	SceneActivity float64         `json:"scene_activity" bigquery:"scene_activity"`
	Objects       json.RawMessage `json:"objects" bigquery:"objects"`
	AudioEvents   json.RawMessage `json:"audio_events" bigquery:"audio_events"`
	KeyframeB64   string          `json:"keyframe_b64"`
	KeyframeGCS   string          `bigquery:"keyframe_gcs"`
}

func (e *VisionEvent) BQTimestamp() time.Time {
	return time.Unix(int64(e.Timestamp), int64((e.Timestamp-float64(int64(e.Timestamp)))*1e9))
}

func main() {
	projectID := envOrDefault("GCP_PROJECT", "")
	subName := envOrDefault("PUBSUB_SUBSCRIPTION", "visionbrain-events-sub")
	bucketName := envOrDefault("GCS_BUCKET", "visionbrain-keyframes")
	bqDataset := envOrDefault("BQ_DATASET", "visionbrain")
	bqTable := envOrDefault("BQ_TABLE", "events")

	if projectID == "" {
		log.Fatal("GCP_PROJECT env required")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("Shutting down…")
		cancel()
	}()

	// Clients
	psClient, err := pubsub.NewClient(ctx, projectID)
	if err != nil {
		log.Fatalf("pubsub client: %v", err)
	}
	defer psClient.Close()

	gcsClient, err := storage.NewClient(ctx)
	if err != nil {
		log.Fatalf("storage client: %v", err)
	}
	defer gcsClient.Close()
	bucket := gcsClient.Bucket(bucketName)

	bqClient, err := bigquery.NewClient(ctx, projectID)
	if err != nil {
		log.Fatalf("bigquery client: %v", err)
	}
	defer bqClient.Close()
	inserter := bqClient.Dataset(bqDataset).Table(bqTable).Inserter()

	sub := psClient.Subscription(subName)
	sub.ReceiveSettings.MaxOutstandingMessages = 100

	log.Printf("Listening on %s…", subName)

	err = sub.Receive(ctx, func(ctx context.Context, msg *pubsub.Message) {
		var evt VisionEvent
		if err := json.Unmarshal(msg.Data, &evt); err != nil {
			log.Printf("bad message: %v", err)
			msg.Ack()
			return
		}

		// Store keyframe in GCS
		if evt.KeyframeB64 != "" {
			data, err := base64.StdEncoding.DecodeString(evt.KeyframeB64)
			if err == nil {
				objName := fmt.Sprintf("%s/%s.jpg", evt.CameraID, evt.EventID)
				w := bucket.Object(objName).NewWriter(ctx)
				w.ContentType = "image/jpeg"
				if _, err := w.Write(data); err == nil {
					w.Close()
					evt.KeyframeGCS = fmt.Sprintf("gs://%s/%s", bucketName, objName)
				} else {
					w.Close()
					log.Printf("GCS write error: %v", err)
				}
			}
			evt.KeyframeB64 = "" // don't store in BQ
		}

		// Insert into BigQuery
		type bqRow struct {
			EventID       string    `bigquery:"event_id"`
			Timestamp     time.Time `bigquery:"timestamp"`
			CameraID      string    `bigquery:"camera_id"`
			EventType     string    `bigquery:"event_type"`
			SceneActivity float64   `bigquery:"scene_activity"`
			Objects       string    `bigquery:"objects"`
			AudioEvents   string    `bigquery:"audio_events"`
			KeyframeGCS   string    `bigquery:"keyframe_gcs"`
		}
		row := &bqRow{
			EventID:       evt.EventID,
			Timestamp:     evt.BQTimestamp(),
			CameraID:      evt.CameraID,
			EventType:     evt.EventType,
			SceneActivity: evt.SceneActivity,
			Objects:       string(evt.Objects),
			AudioEvents:   string(evt.AudioEvents),
			KeyframeGCS:   evt.KeyframeGCS,
		}
		if err := inserter.Put(ctx, row); err != nil {
			log.Printf("BQ insert error: %v", err)
		}

		msg.Ack()
		log.Printf("Processed event %s camera=%s objects=%d", evt.EventID, evt.CameraID, len(evt.Objects))
	})

	if err != nil && ctx.Err() == nil {
		log.Fatalf("receive error: %v", err)
	}
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
