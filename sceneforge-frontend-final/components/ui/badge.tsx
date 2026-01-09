"use client"

interface BadgeProps {
  label: string
  variant?: "primary" | "secondary" | "success" | "warning" | "error"
  size?: "sm" | "md"
}

export function Badge({ label, variant = "primary", size = "md" }: BadgeProps) {
  const variantClasses = {
    primary: "bg-primary/20 text-primary border border-primary/30",
    secondary: "bg-secondary/20 text-secondary border border-secondary/30",
    success: "bg-green-500/20 text-green-400 border border-green-500/30",
    warning: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
    error: "bg-red-500/20 text-red-400 border border-red-500/30",
  }

  const sizeClasses = {
    sm: "px-2 py-1 text-xs",
    md: "px-3 py-1.5 text-sm",
  }

  return <span className={`rounded-full font-medium ${variantClasses[variant]} ${sizeClasses[size]}`}>{label}</span>
}
