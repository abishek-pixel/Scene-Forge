"use client"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Search, Plus, MoreVertical, Eye, Share2, Trash2, Copy } from "lucide-react"
import Link from "next/link"
import { useState, useEffect } from "react"
import { api } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"

interface Scene {
  id: string
  name: string
  status: "processing" | "completed" | "failed" | "cancelled"
  progress: number
  message: string
  stage?: string
  eta?: string
  details?: Array<{
    step: string
    completed: boolean
  }>
  sceneId: string
  createdAt: string
  updatedAt: string
  thumbnail?: string
  size?: string
  tags?: string[]
}

export default function ScenesPage() {
  const [scenes, setScenes] = useState<Scene[]>([])
  const [loading, setLoading] = useState(true)

  // Fetch processing tasks
  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const token = localStorage.getItem('token') || undefined
        const response = await api.processing.list(token)
        setScenes(response.tasks || [])
      } catch (error) {
        console.error('Failed to fetch tasks:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchTasks()
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchTasks, 5000)
    return () => clearInterval(interval)
  }, [])

  const [searchQuery, setSearchQuery] = useState("")
  const [filterStatus, setFilterStatus] = useState("all")
  const [selectedScene, setSelectedScene] = useState<Scene | null>(null)

  const filteredScenes = scenes
    .filter((s) => s.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .filter((s) => (filterStatus === "all" ? true : s.status === filterStatus))

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-green-500 bg-green-500/10"
      case "processing":
        return "text-yellow-500 bg-yellow-500/10"
      case "failed":
        return "text-red-500 bg-red-500/10"
      case "cancelled":
        return "text-orange-500 bg-orange-500/10"
      default:
        return "text-muted-foreground bg-muted/10"
    }
  }

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">My Scenes</h1>
            <p className="text-muted-foreground mt-2">Manage all your 3D scenes and projects.</p>
          </div>
          <Link href="/upload">
            <Button className="btn-primary">
              <Plus className="w-4 h-4 mr-2" />
              New Scene
            </Button>
          </Link>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 flex-wrap">
          <div className="flex-1 min-w-64 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search scenes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>

        {/* Scenes Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredScenes.map((scene) => (
            <Card
              key={scene.id}
              className="glass-panel overflow-hidden hover:border-primary/50 transition-all group cursor-pointer"
              onClick={() => setSelectedScene(scene)}
            >
              {/* Thumbnail */}
              <div className="aspect-video bg-muted relative overflow-hidden group-hover:opacity-80 transition-opacity">
                <div className="absolute inset-0 bg-linear-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-primary/50">3D</div>
                  </div>
                </div>
                <div className="absolute top-2 right-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(scene.status)}`}>
                    {scene.status}
                  </span>
                </div>
              </div>

              {/* Content */}
              <div className="p-4 space-y-3">
                <div>
                  <h3 className="font-semibold text-foreground line-clamp-2">{scene.name}</h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {new Date(scene.createdAt).toLocaleString()}
                  </p>
                </div>

                {/* Progress Info */}
                <div className="space-y-2">
                  {scene.stage && (
                    <p className="text-sm text-muted-foreground">{scene.stage}</p>
                  )}
                  {scene.status === 'processing' && (
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ width: `${scene.progress}%` }}
                      />
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{scene.eta || (scene.status === 'completed' ? 'Completed' : 'Processing...')}</span>
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-2">
                  <Link 
                    href={`/results?sceneId=${scene.sceneId}`}
                    className="flex-1"
                  >
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full bg-card/50 border-border/50 hover:bg-card text-foreground"
                      onClick={(e) => {
                        e.stopPropagation()
                      }}
                    >
                      <Eye className="w-4 h-4 mr-1" />
                      View
                    </Button>
                  </Link>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-card/50 border-border/50 hover:bg-card text-foreground"
                    onClick={(e) => {
                      e.stopPropagation()
                    }}
                  >
                    <MoreVertical className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Scene Details Modal */}
        {selectedScene && (
          <div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedScene(null)}
          >
            <Card
              className="glass-panel max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 space-y-6">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-foreground">{selectedScene.name}</h2>
                    <p className="text-sm text-muted-foreground mt-1">
                      Created: {new Date(selectedScene.createdAt).toLocaleString()}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedScene(null)}
                    className="p-2 hover:bg-muted rounded-lg transition-colors"
                  >
                    ✕
                  </button>
                </div>

                {/* Preview & Progress */}
                <div className="aspect-video bg-muted rounded-lg overflow-hidden relative">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-6xl font-bold text-primary/50">3D</div>
                      <p className="text-muted-foreground mt-2">{selectedScene.message || 'Processing your scene...'}</p>
                    </div>
                  </div>
                  {selectedScene.status === 'processing' && (
                    <div className="absolute bottom-0 left-0 right-0 p-4 bg-background/80 backdrop-blur-sm">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>{selectedScene.stage || 'Processing...'}</span>
                          <span>{selectedScene.progress}%</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div
                            className="bg-primary h-2 rounded-full transition-all"
                            style={{ width: `${selectedScene.progress}%` }}
                          />
                        </div>
                        {selectedScene.eta && (
                          <p className="text-xs text-muted-foreground text-center">{selectedScene.eta}</p>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Processing Steps */}
                {selectedScene.details && (
                  <div className="space-y-4">
                    <h3 className="text-sm font-medium text-foreground">Processing Steps</h3>
                    <div className="space-y-2">
                      {selectedScene.details.map((detail, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <div
                            className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                              detail.completed
                                ? 'bg-primary/20 text-primary'
                                : 'bg-muted text-muted-foreground'
                            }`}
                          >
                            {detail.completed ? '✓' : i + 1}
                          </div>
                          <span
                            className={`text-sm ${
                              detail.completed ? 'text-muted-foreground' : 'text-foreground'
                            }`}
                          >
                            {detail.step}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Info */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Status</p>
                    <p className={`text-lg font-semibold mt-1 ${getStatusColor(selectedScene.status).split(" ")[0]}`}>
                      {selectedScene.status}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Scene ID</p>
                    <p className="text-lg font-semibold text-foreground mt-1">{selectedScene.sceneId}</p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 border-t border-border/30 pt-6">
                  <Link 
                    href={`/results?sceneId=${selectedScene.sceneId}`}
                    className="flex-1"
                  >
                    <Button className="w-full btn-primary">
                      <Eye className="w-4 h-4 mr-2" />
                      View Scene
                    </Button>
                  </Link>
                  <Button
                    variant="outline"
                    className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                  >
                    <Share2 className="w-4 h-4 mr-2" />
                    Share
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                  >
                    <Copy className="w-4 h-4 mr-2" />
                    Duplicate
                  </Button>
                  <Button
                    variant="outline"
                    className="bg-destructive/10 border-destructive/30 hover:bg-destructive/20 text-destructive"
                  >
                    <Trash2 className="w-4 h-4" />
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
