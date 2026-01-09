"use client"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Clock, AlertCircle, Pause, X } from "lucide-react"
import { useState, useEffect } from "react"
import { api } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"

interface ProcessingTask {
  id: string;
  name: string;
  progress: number;
  status: string;
  message: string;
  createdAt: string;
  updatedAt: string;
  stage?: string;
  eta?: string;
  details?: Array<{
    step: string;
    completed: boolean;
  }>;
}

export default function ProcessingPage() {
  const [tasks, setTasks] = useState<ProcessingTask[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  // Fetch active tasks
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('No auth token found');
      return;
    }

    const fetchTasks = async () => {
      try {
        const response = await api.processing.list(token);
        setTasks(response.tasks || []);
      } catch (error) {
        console.error('Failed to fetch tasks:', error);
        toast({
          title: 'Error',
          description: 'Failed to load processing tasks. Please try again.',
          variant: 'destructive',
        });
      } finally {
        setLoading(false);
      }
    };

    fetchTasks();
    
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchTasks, 5000);
    return () => clearInterval(interval);
  }, [])

  const [expandedTask, setExpandedTask] = useState<string | null>(null)

  const pauseTask = async (id: string) => {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('No auth token found');
      return;
    }

    try {
      await api.processing.pause(id, token);
      toast({
        title: 'Task Paused',
        description: 'The processing task has been paused.',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to pause the task. Please try again.',
        variant: 'destructive',
      });
    }
  }

  const cancelTask = async (id: string) => {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('No auth token found');
      return;
    }

    try {
      await api.processing.cancel(id, token);
      setTasks((prev) => prev.filter((task) => task.id !== id));
      toast({
        title: 'Task Cancelled',
        description: 'The processing task has been cancelled.',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to cancel the task. Please try again.',
        variant: 'destructive',
      });
    }
  }

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Processing</h1>
          <p className="text-muted-foreground mt-2">Monitor your active conversion tasks in real-time.</p>
        </div>

        {/* Active Tasks */}
        <div className="space-y-4">
          {tasks.length > 0 ? (
            tasks.map((task) => (
              <Card key={task.id} className="glass-panel overflow-hidden">
                <div
                  className="p-6 space-y-4 cursor-pointer hover:bg-muted/20 transition-colors"
                  onClick={() => setExpandedTask(expandedTask === task.id ? null : task.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="font-semibold text-foreground">{task.name}</h3>
                      <p className="text-sm text-muted-foreground mt-1">{task.stage}</p>
                      <p className="text-xs text-muted-foreground mt-1">Started {new Date(task.createdAt).toLocaleString()}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        {task.eta}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation()
                            pauseTask(task.id)
                          }}
                          className="bg-card/50 border-border/50 hover:bg-card text-foreground"
                        >
                          <Pause className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation()
                            cancelTask(task.id)
                          }}
                          className="bg-destructive/10 border-destructive/30 hover:bg-destructive/20 text-destructive"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>

                    <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">{task.status}</span>
                      <span className="text-sm font-medium text-foreground">{task.progress}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-3">
                      <div
                        className="bg-linear-to-r from-primary to-accent h-3 rounded-full transition-all duration-500"
                        style={{ width: `${task.progress}%` }}
                      ></div>
                    </div>
                  </div>
                  {task.message && (
                    <div className="mt-2 text-sm text-muted-foreground">
                      {task.message}
                    </div>
                  )}
                </div>

                {/* Expanded Details */}
                {expandedTask === task.id && (
                  <div className="border-t border-border/30 p-6 bg-muted/10 space-y-4">
                    <h4 className="font-medium text-foreground">Processing Steps</h4>
                    <div className="space-y-3">
                      {task.details?.map((detail, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <div
                            className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                              detail.completed ? "bg-green-500/20 text-green-500" : "bg-muted text-muted-foreground"
                            }`}
                          >
                            {detail.completed ? "✓" : i + 1}
                          </div>
                          <span
                            className={`text-sm ${
                              detail.completed ? "text-muted-foreground line-through" : "text-foreground"
                            }`}
                          >
                            {detail.step}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Card>
            ))
          ) : (
            <Card className="glass-panel p-6">
              <div className="flex items-center gap-3 text-muted-foreground">
                <AlertCircle className="w-5 h-5" />
                <p>No active tasks. Start by uploading content to begin processing.</p>
              </div>
            </Card>
          )}
        </div>

        {/* Queue */}
        <div>
          <h2 className="text-xl font-bold text-foreground mb-4">Queued Tasks</h2>
          <Card className="glass-panel p-6">
            <div className="flex items-center gap-3 text-muted-foreground">
              <AlertCircle className="w-5 h-5" />
              <p>No queued tasks. All uploads are being processed.</p>
            </div>
          </Card>
        </div>

        {/* Processing Tips */}
        <Card className="glass-panel p-6 border-primary/30 bg-primary/5">
          <h3 className="font-semibold text-foreground mb-3">Processing Tips</h3>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• Processing time depends on file size and selected quality</li>
            <li>• You can pause tasks and resume them later</li>
            <li>• Higher quality settings produce better 3D models but take longer</li>
            <li>• You'll receive an email notification when processing is complete</li>
          </ul>
        </Card>
      </div>
    </AppLayout>
  )
}
