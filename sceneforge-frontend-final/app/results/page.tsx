"use client"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, Eye, Trash2, Share2, Search } from "lucide-react"
import { useState, useEffect } from "react"
import { api } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"

interface SceneResult {
  id: number | string
  name: string
  date: string
  size: string
  format: string
  quality: string
  processingTime: string
  fileCount: number
  files?: Array<{
    name: string
    path: string
    size: number
  }>
}

export default function ResultsPage() {
  const [results, setResults] = useState<SceneResult[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedResult, setSelectedResult] = useState<SceneResult | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortBy, setSortBy] = useState("recent")
  const { toast } = useToast()
  const router = useRouter()

  // Fetch completed scenes from backend
  useEffect(() => {
    const fetchScenes = async () => {
      try {
        const token = localStorage.getItem('token')
        const response = await api.processing.scenes(token || undefined)
        
        // Convert ISO dates to relative dates
        const scenes = response.scenes?.map((scene: any) => ({
          ...scene,
          date: new Date(scene.date).toLocaleDateString() + " ago"
        })) || []
        
        setResults(scenes)
      } catch (error) {
        console.error('Failed to fetch scenes:', error)
        toast({
          title: 'Error',
          description: 'Failed to load completed scenes.',
          variant: 'destructive',
        })
      } finally {
        setLoading(false)
      }
    }

    fetchScenes()
    // Poll for updates every 10 seconds
    const interval = setInterval(fetchScenes, 10000)
    return () => clearInterval(interval)
  }, [toast])

  const filteredResults = results
    .filter((r) => r.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === "recent") return 0
      if (sortBy === "size") return Number.parseInt(b.size) - Number.parseInt(a.size)
      if (sortBy === "name") return a.name.localeCompare(b.name)
      return 0
    })

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Results</h1>
          <p className="text-muted-foreground mt-2">View and manage your completed 3D scenes.</p>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search results..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="recent">Most Recent</option>
            <option value="size">Largest</option>
            <option value="name">Name (A-Z)</option>
          </select>
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {loading ? (
            <Card className="glass-panel p-8 col-span-full text-center">
              <p className="text-muted-foreground">Loading completed scenes...</p>
            </Card>
          ) : filteredResults.length === 0 ? (
            <Card className="glass-panel p-8 col-span-full text-center">
              <p className="text-muted-foreground">No completed scenes yet. Start by uploading content!</p>
            </Card>
          ) : (
            filteredResults.map((result) => (
            <Card
              key={result.id}
              className="glass-panel overflow-hidden hover:border-primary/50 transition-all group cursor-pointer"
              onClick={() => setSelectedResult(result)}
            >
              {/* Thumbnail */}
              <div className="aspect-video bg-muted relative overflow-hidden group-hover:opacity-80 transition-opacity">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-primary/50">3D</div>
                    <p className="text-xs text-primary/40 mt-2">{result.format}</p>
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-4 space-y-3">
                <div>
                  <h3 className="font-semibold text-foreground line-clamp-2">{result.name}</h3>
                  <p className="text-xs text-muted-foreground mt-1">{result.date}</p>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 rounded bg-muted/50">
                    <p className="text-muted-foreground">Size</p>
                    <p className="font-medium text-foreground">{result.size}</p>
                  </div>
                  <div className="p-2 rounded bg-muted/50">
                    <p className="text-muted-foreground">Quality</p>
                    <p className="font-medium text-foreground">{result.quality}</p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                    onClick={(e) => {
                      e.stopPropagation()
                      router.push(`/scenes/${result.id}`)
                    }}
                  >
                    <Eye className="w-4 h-4 mr-1" />
                    View
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                    onClick={(e) => {
                      e.stopPropagation()
                    }}
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Download
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-card/50 border-border/50 hover:bg-destructive/20 text-destructive"
                    onClick={(e) => {
                      e.stopPropagation()
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
            ))
          )}
        </div>

        {/* Details Modal */}
        {selectedResult && (
          <div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedResult(null)}
          >
            <Card
              className="glass-panel max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 space-y-6">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-foreground">{selectedResult.name}</h2>
                    <p className="text-sm text-muted-foreground mt-1">{selectedResult.date}</p>
                  </div>
                  <button
                    onClick={() => setSelectedResult(null)}
                    className="p-2 hover:bg-muted rounded-lg transition-colors"
                  >
                    ✕
                  </button>
                </div>

                {/* Preview */}
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-primary/50">3D</div>
                    <p className="text-muted-foreground mt-2">3D Preview Coming Soon</p>
                  </div>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">File Size</p>
                    <p className="text-lg font-semibold text-foreground mt-1">{selectedResult.size}</p>
                  </div>
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Format</p>
                    <p className="text-lg font-semibold text-foreground mt-1">{selectedResult.format}</p>
                  </div>
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Quality</p>
                    <p className="text-lg font-semibold text-foreground mt-1">{selectedResult.quality}</p>
                  </div>
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Processing Time</p>
                    <p className="text-lg font-semibold text-foreground mt-1">{selectedResult.processingTime}</p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 border-t border-border/30 pt-6">
                  <Button className="flex-1 btn-primary"
                    onClick={() => {
                      if (selectedResult?.files?.[0]) {
                        const file = selectedResult.files[0]
                        // Download the file
                        const link = document.createElement('a')
                        link.href = file.path
                        link.download = file.name
                        link.click()
                      }
                    }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                  >
                    <Share2 className="w-4 h-4 mr-2" />
                    Share
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
