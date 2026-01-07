"use client"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, Share2, ArrowLeft, Loader2 } from "lucide-react"
import { useRouter, useParams } from "next/navigation"
import { useState, useEffect } from "react"
import { api } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"
import { Viewer3D } from "@/components/ui/3d-viewer"

interface SceneFile {
  name: string
  path: string
  size: number
}

interface Scene {
  id: string
  name: string
  date: string
  size: string
  format: string
  quality: string
  processingTime: string
  fileCount: number
  files?: SceneFile[]
  status: string
}

export default function SceneDetailPage() {
  const router = useRouter()
  const params = useParams()
  const sceneId = params.id as string
  
  const [scene, setScene] = useState<Scene | null>(null)
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState(false)
  const [modelUrl, setModelUrl] = useState<string | null>(null)
  const [meshLoading, setMeshLoading] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    const fetchScenes = async () => {
      try {
        const token = localStorage.getItem('token')
        const response = await api.processing.scenes(token || undefined)
        
        // Find the scene with matching ID
        const foundScene = response.scenes?.find((s: any) => s.id === sceneId)
        if (foundScene) {
          setScene(foundScene)
          
          // Build model URL - look for .glb file
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
          if (foundScene.files && foundScene.files.length > 0) {
            // Find GLB file
            const glbFile = foundScene.files.find((f: any) => f.name.endsWith('.glb'))
            if (glbFile) {
              setModelUrl(`${apiUrl}${glbFile.path}`)
              setMeshLoading(false)
            }
          }
        } else {
          toast({
            title: 'Scene not found',
            description: 'The requested scene could not be found.',
            variant: 'destructive',
          })
          router.push('/results')
        }
      } catch (error) {
        console.error('Failed to fetch scene:', error)
        toast({
          title: 'Error',
          description: 'Failed to load scene details.',
          variant: 'destructive',
        })
      } finally {
        setLoading(false)
      }
    }

    fetchScenes()
  }, [sceneId, router, toast])

  const handleDownload = async () => {
    if (!scene?.files || scene.files.length === 0) {
      toast({
        title: 'No files',
        description: 'No files available for download.',
        variant: 'destructive',
      })
      return
    }

    try {
      setDownloading(true)
      // Download the first (primary) file
      const file = scene.files[0]
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const fullUrl = `${apiUrl}${file.path}`
      
      // Open the download URL in a new tab
      window.open(fullUrl, '_blank')
      
      toast({
        title: 'Download started',
        description: `Downloading ${file.name}...`,
      })
    } catch (error) {
      toast({
        title: 'Download failed',
        description: 'Failed to download the scene file.',
        variant: 'destructive',
      })
    } finally {
      setDownloading(false)
    }
  }

  if (loading) {
    return (
      <AppLayout>
        <div className="p-8 flex items-center justify-center min-h-[400px]">
          <div className="text-center space-y-4">
            <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto" />
            <p className="text-muted-foreground">Loading scene details...</p>
          </div>
        </div>
      </AppLayout>
    )
  }

  if (!scene) {
    return (
      <AppLayout>
        <div className="p-8 space-y-8">
          <Button
            variant="ghost"
            onClick={() => router.push('/results')}
            className="text-primary hover:text-primary/80"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Results
          </Button>
          
          <Card className="glass-panel p-8 text-center">
            <p className="text-muted-foreground">Scene not found</p>
          </Card>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => router.push('/results')}
            className="text-primary hover:text-primary/80"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Results
          </Button>
        </div>

        <div className="space-y-6 max-w-4xl">
          {/* Scene Title */}
          <div>
            <h1 className="text-4xl font-bold text-foreground">{scene.name}</h1>
            <p className="text-muted-foreground mt-2">Created: {scene.date}</p>
          </div>

          {/* Preview Area */}
          <Card className="glass-panel overflow-hidden">
            <Viewer3D 
              modelUrl={modelUrl ?? undefined}
              loading={meshLoading || !modelUrl}
              onLoaded={() => setMeshLoading(false)}
            />
          </Card>

          {/* Details Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="glass-panel p-4">
              <p className="text-xs text-muted-foreground">File Size</p>
              <p className="text-lg font-semibold text-foreground mt-1">{scene.size}</p>
            </Card>
            <Card className="glass-panel p-4">
              <p className="text-xs text-muted-foreground">Format</p>
              <p className="text-lg font-semibold text-foreground mt-1">{scene.format}</p>
            </Card>
            <Card className="glass-panel p-4">
              <p className="text-xs text-muted-foreground">Quality</p>
              <p className="text-lg font-semibold text-foreground mt-1">{scene.quality}</p>
            </Card>
            <Card className="glass-panel p-4">
              <p className="text-xs text-muted-foreground">Processing Time</p>
              <p className="text-lg font-semibold text-foreground mt-1">{scene.processingTime}</p>
            </Card>
          </div>

          {/* Files List */}
          {scene.files && scene.files.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-foreground">Output Files</h2>
              <Card className="glass-panel overflow-hidden">
                <div className="divide-y divide-border/30">
                  {scene.files.map((file, index) => (
                    <div key={index} className="p-4 flex items-center justify-between hover:bg-muted/20 transition-colors">
                      <div className="flex-1">
                        <p className="font-medium text-foreground">{file.name}</p>
                        <p className="text-sm text-muted-foreground mt-1">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        className="bg-card/50 border-border/50 hover:bg-card text-foreground"
                        onClick={() => {
                          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
                          const fullUrl = `${apiUrl}${file.path}`
                          window.open(fullUrl, '_blank')
                        }}
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 border-t border-border/30 pt-6">
            <Button 
              className="flex-1 btn-primary"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Downloading...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Download Scene
                </>
              )}
            </Button>
            <Button
              variant="outline"
              className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
            >
              <Share2 className="w-4 h-4 mr-2" />
              Share
            </Button>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
