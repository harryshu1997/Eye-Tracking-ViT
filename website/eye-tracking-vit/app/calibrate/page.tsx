"use client"
import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"

export default function GazeGamePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [gridPoints, setGridPoints] = useState<{ x: number, y: number }[]>([])
  const [index, setIndex] = useState(0)
  const router = useRouter()
  const [status, setStatus] = useState("Waiting...")

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    })

    const marginX = 50
    const marginY = 50
    const rows = 4, cols = 4
    const points = []
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x = marginX + (window.innerWidth - 2 * marginX) * col / (cols - 1)
        const y = marginY + (window.innerHeight - 2 * marginY) * row / (rows - 1)
        points.push({ x, y })
      }
    }
    setGridPoints(points)
  }, [])

  const captureAndSend = async () => {
    if (!videoRef.current || index >= gridPoints.length) return
    const canvas = document.createElement("canvas")
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0)

    const blob = await new Promise<Blob>((resolve) =>
      canvas.toBlob((blob) => blob && resolve(blob), "image/jpeg")
    )

    const { x, y } = gridPoints[index]
    const formData = new FormData()
    formData.append("file", blob, `image${index}.jpg`)
    formData.append("x", String(Math.round(x)))
    formData.append("y", String(Math.round(y)))
    formData.append("screen_width", String(window.innerWidth))
    formData.append("screen_height", String(window.innerHeight))
    formData.append("index", String(index))

    await fetch("http://localhost:8000/save_click", {
      method: "POST",
      body: formData,
    })

    setIndex((i) => i + 1)
    if (index + 1 >= gridPoints.length) {
        setStatus("Done!")
        router.push("/")
      } else {
        setStatus(`Click dot ${index + 1}`)
      }
  }

  const currentDot = gridPoints[index] || null

  return (
    <div className="w-full h-screen bg-black relative">
      <video ref={videoRef} autoPlay className="hidden" />
      {currentDot && (
        <div 
          style={{
            position: "absolute",
            left: `${currentDot.x - 15}px`,
            top: `${currentDot.y - 15}px`,
            width: "30px",
            height: "30px",
            backgroundColor: "red",
            borderRadius: "50%",
            zIndex: 10,
          }}
          onClick={captureAndSend}
        />
      )}
      <p className="absolute top-4 left-4 text-white">{status}</p>
    </div>
  )
}
