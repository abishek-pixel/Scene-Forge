"use client"

import type React from "react"

import { useState, useRef } from "react"
import { LoadingSpinner } from "./loading-spinner"

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void
  accept?: string
  multiple?: boolean
  maxSize?: number
}

export function FileUpload({
  onFilesSelected,
  accept = "*",
  multiple = true,
  maxSize = 50 * 1024 * 1024,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    processFiles(files)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    processFiles(files)
  }

  const processFiles = (files: File[]) => {
    const validFiles = files.filter((file) => file.size <= maxSize)
    if (validFiles.length > 0) {
      setIsLoading(true)
      setTimeout(() => {
        onFilesSelected(validFiles)
        setIsLoading(false)
      }, 500)
    }
  }

  return (
    <div
      className={`glass-panel p-8 border-2 border-dashed transition-all duration-200 cursor-pointer ${
        isDragging ? "border-primary bg-primary/10" : "border-border/30"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleFileSelect}
        className="hidden"
      />

      <div className="flex flex-col items-center justify-center">
        {isLoading ? (
          <>
            <LoadingSpinner size="lg" />
            <p className="mt-4 text-foreground font-medium">Processing files...</p>
          </>
        ) : (
          <>
            <svg className="w-12 h-12 text-primary mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="text-foreground font-medium">Drag and drop files here</p>
            <p className="text-muted-foreground text-sm mt-1">or click to select</p>
            <p className="text-muted-foreground text-xs mt-2">Max file size: {maxSize / (1024 * 1024)}MB</p>
          </>
        )}
      </div>
    </div>
  )
}
