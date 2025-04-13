"use client"
import { useEffect, useRef } from "react";
export default function Page() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return 
    // Get webcam stream
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    })

    const sendFrame = async() => {
      if (!videoRef.current) {
        return
      }
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0)
    
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((blob) => {
          if (blob) {
            resolve(blob)
          } else {
            reject(new Error("Failed to create blob"))
          }
        }, "image/jpeg")
      )
      const formData = new FormData()
      formData.append("file", blob, "eye.jpg")
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })
      const { x, y, direction } = await res.json()
      if (cursorRef.current) {
        cursorRef.current.style.left = `${x}px`
        cursorRef.current.style.top = `${y}px`
      }
      console.log(x, y, direction)
    }
    const interval = setInterval(sendFrame, 1000) 
    return () => clearInterval(interval)
  }, []);

  return (
    <div className="flex flex-col items-center justify-center ">
      <h1 className="text-xl font-bold">Eye Tracking ViT Project</h1>
      <video ref={videoRef} autoPlay style={{ display: 'none' }} />
      <div
        ref={cursorRef}
        style={{
          position: 'absolute',
          width: '15px',
          height: '15px',
          background: 'green',
          borderRadius: '50%',
          pointerEvents: 'none',
        }}
      />
    </div>
    
);
} 