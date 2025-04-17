"use client"
import { useEffect, useRef } from "react";
export default function Page() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const currentPos = useRef({ x: 0, y: 0 });
  const targetPos = useRef({ x: 0, y: 0 });
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
      formData.append("screen_width", window.innerWidth.toString())
      formData.append("screen_height", window.innerHeight.toString())
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })
      const { x, y, direction } = await res.json()
      targetPos.current = {x, y}
      console.log(x, y, direction)

    }
    const animateCursor = () => {
    if (cursorRef.current) {
      // LERP = linear interpolation
      const lerp = (start : number, end : number, t : number) => start + (end - start) * t

      currentPos.current.x = lerp(currentPos.current.x, targetPos.current.x, 0.05)
      currentPos.current.y = lerp(currentPos.current.y, targetPos.current.y, 0.05)

      cursorRef.current.style.left = `${currentPos.current.x}px`
      cursorRef.current.style.top = `${currentPos.current.y}px`
      }

    requestAnimationFrame(animateCursor)
    }
    requestAnimationFrame(animateCursor)
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