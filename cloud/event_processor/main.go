// Package main implements the cloud event processor.
// Consumes events from Pub/Sub, stores keyframes in GCS, metadata in BigQuery,
// and fans out to downstream consumers.
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
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

// ── Metrics ──────────────────────────────────────────────────────────
type metrics struct {
	processed atomic.Int64
	errors    atomic.Int64
	bqRows    atomic.Int64
	gcsWrites atomic.Int64
}

func (m *metrics) log() {
	slog.Info("metrics",
		"processed", m.processed.Load(),
		"errors", m.errors.Load(),
		"bq_rows", m.bqRows.Load(),
		"gcs_writes", m.gcsWrites.Load(),
	)
}

// ── Batch BigQuery Inserter ─────────────────────────────────────────
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

type batchInserter struct {
	inserter *bigquery.Inserter
	mu       sync.Mutex
	buf      []*bqRow
	maxBatch int
	flushInt time.Duration
	m        *metrics
}

func newBatchInserter(ins *bigquery.Inserter, m *metrics) *batchInserter {
	return &batchInserter{
		inserter: ins,
		maxBatch: 50,
		flushInt: 2 * time.Second,
		m:        m,
	}
}

func (b *batchInserter) add(row *bqRow) {
	b.mu.Lock()
	b.buf = append(b.buf, row)
	shouldFlush := len(b.buf) >= b.maxBatch
	b.mu.Unlock()
	if shouldFlush {
		b.flush(context.Background())
	}
}

func (b *batchInserter) flush(ctx context.Context) {
	b.mu.Lock()
	if len(b.buf) == 0 {
		b.mu.Unlock()
		return
	}
	rows := b.buf
	b.buf = nil
	b.mu.Unlock()

	// Retry with exponential backoff (max 3 attempts)
	for attempt := 0; attempt < 3; attempt++ {
		if err := b.inserter.Put(ctx, rows); err != nil {
			slog.Warn("BQ batch insert failed", "attempt", attempt+1, "rows", len(rows), "error", err)
			b.m.errors.Add(1)
			time.Sleep(time.Duration(1<<attempt) * 500 * time.Millisecond)
			continue
		}
		b.m.bqRows.Add(int64(len(rows)))
		slog.Debug("BQ batch inserted", "rows", len(rows))
		return
	}
	slog.Error("BQ batch insert failed after retries", "dropped_rows", len(rows))
}

func (b *batchInserter) run(ctx context.Context) {
	ticker := time.NewTicker(b.flushInt)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			b.flush(context.Background()) // drain remaining
			return
		case <-ticker.C:
			b.flush(ctx)
		}
	}
}

func main() {
	// Structured logging
	logLevel := slog.LevelInfo
	if envOrDefault("LOG_LEVEL", "") == "debug" {
		logLevel = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel})))

	projectID := envOrDefault("GCP_PROJECT", "")
	subName := envOrDefault("PUBSUB_SUBSCRIPTION", "visionbrain-events-sub")
	bucketName := envOrDefault("GCS_BUCKET", "visionbrain-keyframes")
	bqDataset := envOrDefault("BQ_DATASET", "visionbrain")
	bqTable := envOrDefault("BQ_TABLE", "events")

	if projectID == "" {
		slog.Error("GCP_PROJECT env required")
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		slog.Info("Shutdown signal received")
		cancel()
	}()

	// Clients
	psClient, err := pubsub.NewClient(ctx, projectID)
	if err != nil {
		slog.Error("pubsub client init failed", "error", err)
		os.Exit(1)
	}
	defer psClient.Close()

	gcsClient, err := storage.NewClient(ctx)
	if err != nil {
		slog.Error("storage client init failed", "error", err)
		os.Exit(1)
	}
	defer gcsClient.Close()
	bucket := gcsClient.Bucket(bucketName)

	bqClient, err := bigquery.NewClient(ctx, projectID)
	if err != nil {
		slog.Error("bigquery client init failed", "error", err)
		os.Exit(1)
	}
	defer bqClient.Close()
	inserter := bqClient.Dataset(bqDataset).Table(bqTable).Inserter()

	m := &metrics{}
	batch := newBatchInserter(inserter, m)
	go batch.run(ctx)

	// Periodic metrics logging
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				m.log()
			}
		}
	}()

	sub := psClient.Subscription(subName)
	sub.ReceiveSettings.MaxOutstandingMessages = 100

	slog.Info("Event processor started", "subscription", subName, "bucket", bucketName)

	err = sub.Receive(ctx, func(ctx context.Context, msg *pubsub.Message) {
		var evt VisionEvent
		if err := json.Unmarshal(msg.Data, &evt); err != nil {
			slog.Warn("Malformed message", "error", err, "data_len", len(msg.Data))
			m.errors.Add(1)
			msg.Ack()
			return
		}

		// Store keyframe in GCS with retry
		if evt.KeyframeB64 != "" {
			data, err := base64.StdEncoding.DecodeString(evt.KeyframeB64)
			if err == nil {
				objName := fmt.Sprintf("%s/%s.jpg", evt.CameraID, evt.EventID)
				for attempt := 0; attempt < 3; attempt++ {
					w := bucket.Object(objName).NewWriter(ctx)
					w.ContentType = "image/jpeg"
					if _, err := w.Write(data); err == nil {
						if err := w.Close(); err == nil {
							evt.KeyframeGCS = fmt.Sprintf("gs://%s/%s", bucketName, objName)
							m.gcsWrites.Add(1)
							break
						}
					}
					w.Close()
					slog.Warn("GCS write retry", "attempt", attempt+1, "error", err)
					m.errors.Add(1)
					time.Sleep(time.Duration(1<<attempt) * 200 * time.Millisecond)
				}
			}
			evt.KeyframeB64 = "" // don't store in BQ
		}

		// Batch insert into BigQuery
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
		batch.add(row)

		msg.Ack()
		m.processed.Add(1)
		slog.Debug("Processed event", "event_id", evt.EventID, "camera", evt.CameraID)
	})

	if err != nil && ctx.Err() == nil {
		slog.Error("Receive error", "error", err)
		os.Exit(1)
	}
	// Final flush and metrics
	batch.flush(context.Background())
	m.log()
	slog.Info("Event processor stopped")
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
