"use client"

interface ProgressBarProps {
  value: number
  max?: number
  showLabel?: boolean
  animated?: boolean
}

export function ProgressBar({ value, max = 100, showLabel = true, animated = true }: ProgressBarProps) {
  const percentage = (value / max) * 100

  return (
    <div className="w-full">
      <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full bg-gradient-to-r from-primary to-accent transition-all duration-300 ${
            animated ? "animate-pulse" : ""
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && <p className="text-sm text-muted-foreground mt-1">{Math.round(percentage)}%</p>}
    </div>
  )
}
