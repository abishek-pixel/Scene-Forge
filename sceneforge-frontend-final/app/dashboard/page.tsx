"use client"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, ImageIcon, Eye, Zap, Clock, CheckCircle } from "lucide-react"
import Link from "next/link"
import { useUser } from "@/lib/user-context"

export default function Dashboard() {
  const { user } = useUser()

  if (user?.isNewUser) {
    return (
      <AppLayout>
        <div className="p-8 space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold text-foreground">Welcome, {user.firstName}!</h1>
            <p className="text-muted-foreground mt-2">
              Let's create your first 3D scene. Start by uploading a video or image set.
            </p>
          </div>

          {/* Empty State */}
          <div className="flex flex-col items-center justify-center py-16 space-y-6">
            <div className="w-24 h-24 rounded-lg bg-primary/20 flex items-center justify-center">
              <Zap className="w-12 h-12 text-primary" />
            </div>
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold text-foreground">No scenes yet</h2>
              <p className="text-muted-foreground max-w-md">
                Upload your first video or image set to get started with creating amazing 3D scenes.
              </p>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-md">
              <Link href="/upload">
                <Card className="glass-panel p-6 hover:border-primary/50 transition-all cursor-pointer group h-full">
                  <div className="flex flex-col items-center gap-3 text-center">
                    <div className="p-3 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                      <Upload className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">Upload Video</h3>
                      <p className="text-xs text-muted-foreground mt-1">Convert video to 3D</p>
                    </div>
                  </div>
                </Card>
              </Link>

              <Link href="/upload">
                <Card className="glass-panel p-6 hover:border-primary/50 transition-all cursor-pointer group h-full">
                  <div className="flex flex-col items-center gap-3 text-center">
                    <div className="p-3 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                      <ImageIcon className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">Upload Images</h3>
                      <p className="text-xs text-muted-foreground mt-1">Create from image set</p>
                    </div>
                  </div>
                </Card>
              </Link>
            </div>
          </div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground mt-2">
            Welcome back, {user?.firstName}! Continue creating your 3D scenes.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link href="/upload">
            <Card className="glass-panel p-6 hover:border-primary/50 transition-all cursor-pointer group h-full">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                  <Upload className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Upload Video</h3>
                  <p className="text-sm text-muted-foreground">Convert video to 3D</p>
                </div>
              </div>
            </Card>
          </Link>

          <Link href="/upload">
            <Card className="glass-panel p-6 hover:border-primary/50 transition-all cursor-pointer group h-full">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                  <ImageIcon className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Upload Images</h3>
                  <p className="text-sm text-muted-foreground">Create from image set</p>
                </div>
              </div>
            </Card>
          </Link>

          <Link href="/scenes">
            <Card className="glass-panel p-6 hover:border-primary/50 transition-all cursor-pointer group h-full">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                  <Eye className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">My Scenes</h3>
                  <p className="text-sm text-muted-foreground">View all projects</p>
                </div>
              </div>
            </Card>
          </Link>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="glass-panel p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Scenes</p>
                <p className="text-3xl font-bold text-foreground mt-2">0</p>
              </div>
              <div className="p-3 rounded-lg bg-primary/20">
                <Zap className="w-6 h-6 text-primary" />
              </div>
            </div>
          </Card>

          <Card className="glass-panel p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Processing</p>
                <p className="text-3xl font-bold text-foreground mt-2">0</p>
              </div>
              <div className="p-3 rounded-lg bg-accent/20">
                <Clock className="w-6 h-6 text-accent" />
              </div>
            </div>
          </Card>

          <Card className="glass-panel p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Completed</p>
                <p className="text-3xl font-bold text-foreground mt-2">0</p>
              </div>
              <div className="p-3 rounded-lg bg-green-500/20">
                <CheckCircle className="w-6 h-6 text-green-500" />
              </div>
            </div>
          </Card>
        </div>

        {/* Active Tasks - Empty State */}
        <div>
          <h2 className="text-xl font-bold text-foreground mb-4">Active Tasks</h2>
          <Card className="glass-panel p-8 text-center">
            <p className="text-muted-foreground">No active tasks. Upload content to get started!</p>
          </Card>
        </div>

        {/* Recent Scenes - Empty State */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-foreground">Recent Scenes</h2>
            <Link href="/scenes">
              <Button variant="ghost" className="text-primary hover:text-primary/80">
                View All
              </Button>
            </Link>
          </div>
          <Card className="glass-panel p-8 text-center">
            <p className="text-muted-foreground">No scenes yet. Create your first one to see it here!</p>
          </Card>
        </div>
      </div>
    </AppLayout>
  )
}
