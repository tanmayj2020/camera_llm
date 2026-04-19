"use client";

import { useEffect, useRef } from "react";

interface HLSPlayerProps {
  src: string;
  poster?: string;
  className?: string;
  muted?: boolean;
  autoPlay?: boolean;
}

/**
 * HLS.js video player — renders live CCTV streams via HLS protocol.
 * Falls back to native <video> for browsers with built-in HLS support (Safari).
 */
export default function HLSPlayer({
  src,
  poster,
  className = "",
  muted = true,
  autoPlay = true,
}: HLSPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<any>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    // Check if the browser supports HLS natively (Safari)
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = src;
      if (autoPlay) video.play().catch(() => {});
      return;
    }

    // Use HLS.js for other browsers
    let hls: any;
    import("hls.js").then((HlsModule) => {
      const Hls = HlsModule.default;
      if (!Hls.isSupported()) return;

      hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        liveSyncDurationCount: 3,
        liveMaxLatencyDurationCount: 6,
      });
      hls.loadSource(src);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (autoPlay) video.play().catch(() => {});
      });
      hls.on(Hls.Events.ERROR, (_: any, data: any) => {
        if (data.fatal) {
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            // Retry after network error
            setTimeout(() => hls.startLoad(), 3000);
          } else {
            hls.destroy();
          }
        }
      });
      hlsRef.current = hls;
    });

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [src, autoPlay]);

  return (
    <video
      ref={videoRef}
      className={`w-full h-full object-cover rounded-lg bg-black ${className}`}
      muted={muted}
      playsInline
      poster={poster}
      controls
    />
  );
}
