"use client"

import type { ReactNode } from "react"

interface CardGridProps {
  children: ReactNode
  columns?: 1 | 2 | 3 | 4
}

export function CardGrid({ children, columns = 3 }: CardGridProps) {
  const colClasses = {
    1: "grid-cols-1",
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
  }

  return <div className={`grid ${colClasses[columns]} gap-4`}>{children}</div>
}
