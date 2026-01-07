"use client"

import type React from "react"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Film, ImageIcon, X } from "lucide-react"
import { useState, useRef } from "react"
import { api } from "@/lib/api-client"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"

export default function UploadPage() {
  const [uploadType, setUploadType] = useState<"video" | "images" | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [sceneName, setSceneName] = useState("")
  const [sceneDescription, setSceneDescription] = useState("")
  const [quality, setQuality] = useState("high")
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const { toast } = useToast()
  const router = useRouter()

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const droppedFiles = Array.from(e.dataTransfer.files)
    setFiles((prev) => [...prev, ...droppedFiles])
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files)
      setFiles((prev) => [...prev, ...selectedFiles])
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (!sceneName.trim()) {
      toast({
        title: "Scene name required",
        description: "Please provide a name for your scene.",
        variant: "destructive"
      })
      return
    }
    if (files.length === 0) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload.",
        variant: "destructive"
      })
      return
    }

    try {
      setUploading(true)
      setProgress(0)

      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })
      formData.append('scene_name', sceneName)
      formData.append('prompt', sceneDescription)
      formData.append('quality', quality)

      // Show initial upload status
      setProgress(5);
      const data = await api.processing.upload(formData)
      
      // Upload complete, show processing started
      setProgress(15);
      toast({
        title: "Upload successful",
        description: "File uploaded, starting processing...",
      });
      
      // Start polling for progress
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.processing.status(data.id)
          setProgress(status.progress)

          if (status.status === 'completed') {
            clearInterval(pollInterval)
            toast({
              title: "Processing completed",
              description: "Your scene has been successfully processed!"
            })
            router.push('/scenes')
          } else if (status.status === 'failed') {
            clearInterval(pollInterval)
            throw new Error(status.message || 'Processing failed')
          }
        } catch (error) {
          clearInterval(pollInterval)
          const e = error as Error
          toast({
            title: "Error",
            description: e?.message || String(error),
            variant: "destructive"
          })
        }
      }, 2000)

    } catch (error) {
      const e = error as Error
      toast({
        title: "Upload failed",
        description: e?.message || String(error),
        variant: "destructive"
      })
    } finally {
      setUploading(false)
    }
  }

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Upload Content</h1>
          <p className="text-muted-foreground mt-2">Choose how you want to create your 3D scene.</p>
        </div>

        {!uploadType ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl">
            {/* Video Upload */}
            <Card
              onClick={() => setUploadType("video")}
              className="glass-panel p-8 hover:border-primary/50 transition-all cursor-pointer group"
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="p-4 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                  <Film className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Upload Video</h3>
                  <p className="text-sm text-muted-foreground mt-2">
                    Convert a video file into an interactive 3D scene
                  </p>
                </div>
                <Button className="btn-primary mt-4">Get Started</Button>
              </div>
            </Card>

            {/* Image Upload */}
            <Card
              onClick={() => setUploadType("images")}
              className="glass-panel p-8 hover:border-primary/50 transition-all cursor-pointer group"
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="p-4 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                  <ImageIcon className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Upload Images</h3>
                  <p className="text-sm text-muted-foreground mt-2">Create a 3D scene from a series of images</p>
                </div>
                <Button className="btn-primary mt-4">Get Started</Button>
              </div>
            </Card>
          </div>
        ) : (
          <div className="max-w-3xl">
            <Button
              variant="ghost"
              onClick={() => {
                setUploadType(null)
                setFiles([])
                setSceneName("")
                setSceneDescription("")
              }}
              className="mb-6 text-primary hover:text-primary/80"
            >
              ← Back
            </Button>

            <Card className="glass-panel p-8">
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">
                    {uploadType === "video" ? "Upload Video" : "Upload Images"}
                  </h2>
                  <p className="text-muted-foreground mt-2">
                    {uploadType === "video"
                      ? "Select a video file to convert to 3D"
                      : "Select multiple images to create a 3D scene"}
                  </p>
                </div>

                {/* Upload Area */}
                <div
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  className={`border-2 border-dashed rounded-lg p-12 text-center transition-all cursor-pointer ${
                    dragActive ? "border-primary/50 bg-primary/5" : "border-border/50 hover:border-primary/50"
                  }`}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple={uploadType === "images"}
                    onChange={handleFileSelect}
                    className="hidden"
                    accept={uploadType === "video" ? "video/*" : "image/*"}
                  />
                  <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-foreground font-medium">Drag and drop your files here</p>
                  <p className="text-sm text-muted-foreground mt-2">or click to browse</p>
                  <p className="text-xs text-muted-foreground mt-4">
                    {uploadType === "video"
                      ? "Supported formats: MP4, MOV, AVI (Max 2GB)"
                      : "Supported formats: JPG, PNG, WebP (Max 50 images)"}
                  </p>
                </div>

                {/* File List */}
                {files.length > 0 && (
                  <div className="space-y-2">
                    <h3 className="font-medium text-foreground">Selected Files ({files.length})</h3>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {files.map((file, i) => (
                        <div key={i} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
                            <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                          </div>
                          <button
                            onClick={() => removeFile(i)}
                            className="ml-2 p-1 hover:bg-muted rounded transition-colors"
                          >
                            <X className="w-4 h-4 text-muted-foreground" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Settings */}
                <div className="space-y-4 border-t border-border/30 pt-6">
                  <div>
                    <label className="text-sm font-medium text-foreground">Scene Name</label>
                    <input
                      type="text"
                      placeholder="My 3D Scene"
                      value={sceneName}
                      onChange={(e) => setSceneName(e.target.value)}
                      className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground">Describe your scene or style</label>
                    <p className="text-xs text-muted-foreground mt-1">
                      Help guide how your 3D scene should be stylized and generated
                    </p>
                    <textarea
                      placeholder="Foggy cyberpunk alley with neon lights, futuristic architecture, rain reflections..."
                      value={sceneDescription}
                      onChange={(e) => setSceneDescription(e.target.value)}
                      rows={4}
                      className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground">Quality</label>
                    <select
                      value={quality}
                      onChange={(e) => setQuality(e.target.value)}
                      className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                    >
                      <option value="high">High (Recommended) - Slower processing</option>
                      <option value="medium">Medium - Balanced</option>
                      <option value="low">Low - Faster processing</option>
                    </select>
                  </div>

                  <div className="p-4 rounded-lg bg-primary/5 border border-primary/20">
                    <p className="text-sm text-foreground">
                      <span className="font-medium">Estimated processing time:</span>
                      <span className="text-muted-foreground ml-2">
                        {quality === "high" ? "15-30 minutes" : quality === "medium" ? "10-15 minutes" : "5-10 minutes"}
                      </span>
                    </p>
                  </div>
                </div>

                {uploading && (
                  <div className="space-y-2 border-t border-border/30 pt-6">
                    <Progress value={progress} className="w-full" />
                    <p className="text-sm text-center text-muted-foreground">
                      {progress}% - Processing your scene...
                    </p>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setUploadType(null)
                      setFiles([])
                      setSceneName("")
                      setSceneDescription("")
                    }}
                    className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                  >
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleUpload} 
                    className="flex-1 btn-primary"
                    disabled={uploading || files.length === 0}
                  >
                    {uploading ? 'Processing...' : 'Start Processing'}
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </AppLayout>
  )
}
