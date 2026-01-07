"use client"

import { useState, useEffect } from "react"

export type ToastType = "success" | "error" | "info" | "warning"

interface ToastProps {
  message: string
  type?: ToastType
  duration?: number
  onClose?: () => void
}

export function Toast({ message, type = "info", duration = 3000, onClose }: ToastProps) {
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false)
      onClose?.()
    }, duration)

    return () => clearTimeout(timer)
  }, [duration, onClose])

  if (!isVisible) return null

  const bgColors = {
    success: "bg-green-500/20 border-green-500/30",
    error: "bg-red-500/20 border-red-500/30",
    info: "bg-blue-500/20 border-blue-500/30",
    warning: "bg-yellow-500/20 border-yellow-500/30",
  }

  const textColors = {
    success: "text-green-400",
    error: "text-red-400",
    info: "text-blue-400",
    warning: "text-yellow-400",
  }

  const icons = {
    success: "✓",
    error: "✕",
    info: "ℹ",
    warning: "⚠",
  }

  return (
    <div
      className={`fixed bottom-4 right-4 glass-panel border px-4 py-3 rounded-lg animate-in slide-in-from-bottom-4 ${bgColors[type]}`}
    >
      <div className="flex items-center gap-3">
        <span className={`text-lg font-bold ${textColors[type]}`}>{icons[type]}</span>
        <p className="text-foreground text-sm">{message}</p>
      </div>
    </div>
  )
}
