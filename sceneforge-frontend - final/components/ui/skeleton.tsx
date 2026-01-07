"use client"

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return <div className={`bg-muted/50 rounded-lg animate-pulse ${className}`} />
}
