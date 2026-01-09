"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Upload, Grid, BookOpen, User, LogOut } from "lucide-react"

interface SidebarProps {
  open: boolean
}

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: Grid },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/scenes", label: "My Scenes", icon: Grid },
  { href: "/docs", label: "Docs", icon: BookOpen },
  { href: "/profile", label: "Profile", icon: User },
]

export default function Sidebar({ open }: SidebarProps) {
  const pathname = usePathname()

  return (
    <aside
      className={`${open ? "w-64" : "w-20"} bg-sidebar border-r border-sidebar-border transition-all duration-300 flex flex-col`}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-center border-b border-sidebar-border">
        <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-sidebar-primary/20 border border-sidebar-primary/30">
          <span className="text-lg font-bold text-sidebar-primary">◆</span>
        </div>
        {open && <span className="ml-3 font-bold text-sidebar-foreground">SceneForge</span>}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 ${
                isActive
                  ? "bg-sidebar-primary/20 text-sidebar-primary border border-sidebar-primary/30"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/10"
              }`}
              title={!open ? item.label : undefined}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {open && <span className="text-sm font-medium">{item.label}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Logout */}
      <div className="p-3 border-t border-sidebar-border">
        <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sidebar-foreground hover:bg-sidebar-accent/10 transition-all duration-200">
          <LogOut className="w-5 h-5 flex-shrink-0" />
          {open && <span className="text-sm font-medium">Logout</span>}
        </button>
      </div>
    </aside>
  )
}
