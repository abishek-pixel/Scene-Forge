"use client"

import { Menu, Settings, User } from "lucide-react"
import { Button } from "@/components/ui/button"

interface HeaderProps {
  onMenuClick: () => void
}

export default function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="h-16 border-b border-border/30 bg-card/50 backdrop-blur-md flex items-center justify-between px-6">
      <button onClick={onMenuClick} className="p-2 hover:bg-muted rounded-lg transition-colors">
        <Menu className="w-5 h-5" />
      </button>

      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" className="hover:bg-muted">
          <Settings className="w-5 h-5" />
        </Button>
        <Button variant="ghost" size="icon" className="hover:bg-muted">
          <User className="w-5 h-5" />
        </Button>
      </div>
    </header>
  )
}
