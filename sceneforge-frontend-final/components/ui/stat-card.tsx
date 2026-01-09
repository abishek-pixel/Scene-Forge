"use client"

import type { ReactNode } from "react"

interface StatCardProps {
  label: string
  value: string | number
  icon?: ReactNode
  trend?: "up" | "down"
  trendValue?: string
}

export function StatCard({ label, value, icon, trend, trendValue }: StatCardProps) {
  return (
    <div className="glass-panel p-6 hover:bg-card/60 transition-all duration-200">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-2">{label}</p>
          <p className="text-2xl font-bold text-foreground">{value}</p>
          {trendValue && (
            <p className={`text-xs mt-2 ${trend === "up" ? "text-green-400" : "text-red-400"}`}>
              {trend === "up" ? "↑" : "↓"} {trendValue}
            </p>
          )}
        </div>
        {icon && <div className="text-primary opacity-50">{icon}</div>}
      </div>
    </div>
  )
}
