'use client';

import React, { useEffect, useRef, useState } from 'react';
import { VideoOff } from 'lucide-react';
import { useInterview } from '@/context/InterviewContext';
import { PALETTE } from '@/lib/motion';

export const SelfViewCamera: React.FC = () => {
  const { isCameraActive, setIsCameraActive, candidate } = useInterview();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [streamError, setStreamError] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    if (isCameraActive) {
      navigator.mediaDevices
        ?.getUserMedia({ video: true, audio: false })
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = s;
          }
          setStreamError(false);
        })
        .catch(() => {
          setStreamError(true);
        });
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        const currentStream = videoRef.current.srcObject as MediaStream;
        currentStream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isCameraActive]);

  if (!isCameraActive) return null;

  return (
    <div
      className="absolute top-20 right-8 z-30 w-48 h-36 rounded-2xl overflow-hidden shadow-xl"
      style={{ border: `1px solid ${PALETTE.border}`, backgroundColor: PALETTE.surface }}
    >
      {streamError ? (
        <div className="w-full h-full flex flex-col items-center justify-center p-3 text-center" style={{ color: PALETTE.textSecondary }}>
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center mb-1.5 text-xs font-semibold"
            style={{ backgroundColor: 'rgba(255,255,255,0.06)', color: PALETTE.textPrimary }}
          >
            AN
          </div>
          <p className="text-[10px]">Self-view preview</p>
        </div>
      ) : (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover scale-x-[-1]"
        />
      )}

      {/* Candidate label tag */}
      <div className="absolute bottom-2 left-2 right-2 px-2 py-1 rounded-lg bg-black/60 backdrop-blur-md flex items-center justify-between text-[10px] text-white">
        <span className="truncate">{candidate.name} (You)</span>
        <button
          onClick={() => setIsCameraActive(false)}
          className="text-neutral-400 hover:text-white"
          title="Turn off self view"
        >
          <VideoOff className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
};
