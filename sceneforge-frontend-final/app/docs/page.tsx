"use client"

import { useState } from "react"
import AppLayout from "@/components/layout/app-layout"
import { ChevronDown, Search } from "lucide-react"

interface DocSection {
  id: string
  title: string
  content: string
  subsections?: { id: string; title: string; content: string }[]
}

const docSections: DocSection[] = [
  {
    id: "getting-started",
    title: "Getting Started",
    content:
      "Welcome to SceneForge! This guide will help you get started with creating amazing 3D scenes from your videos and images.",
    subsections: [
      {
        id: "account-setup",
        title: "Account Setup",
        content:
          "Create your account by signing up with your email and phone number. You'll receive an OTP for phone verification. Once verified, you can start creating 3D scenes immediately.",
      },
      {
        id: "first-upload",
        title: "Your First Upload",
        content:
          "Navigate to the Upload section and select your video or image. Add a scene description to guide the AI on how you want your 3D scene to be stylized. Click Process to start the conversion.",
      },
    ],
  },
  {
    id: "uploading-content",
    title: "Uploading Content",
    content: "Learn how to upload and process your videos and images.",
    subsections: [
      {
        id: "supported-formats",
        title: "Supported Formats",
        content:
          "Videos: MP4, MOV (max 30 seconds, 50MB)\nImages: PNG, JPG (max 50MB)\nYou can upload multiple images as a batch for processing.",
      },
      {
        id: "scene-description",
        title: "Scene Description",
        content:
          'Provide a detailed description of how you want your 3D scene to be stylized. Examples:\n- "Foggy cyberpunk alley with neon lights"\n- "Minimalist modern office with warm lighting"\n- "Dark fantasy dungeon with torches"\nThe better your description, the better the AI can generate your scene.',
      },
      {
        id: "quality-settings",
        title: "Quality Settings",
        content:
          "Choose between Standard (faster processing) and High Quality (more detailed results). Processing time varies based on file size and quality setting.",
      },
    ],
  },
  {
    id: "processing",
    title: "Processing & Status",
    content: "Understand how your content is processed.",
    subsections: [
      {
        id: "processing-stages",
        title: "Processing Stages",
        content:
          "1. Analyzing Input - Examining your video/image\n2. Extracting Objects - Identifying 3D objects and elements\n3. Generating Textures - Creating realistic textures and materials\n4. Baking Animations - Processing motion and animations\n5. Finalizing Export - Preparing your 3D scene for download",
      },
      {
        id: "monitoring-progress",
        title: "Monitoring Progress",
        content:
          "View real-time progress on the Processing page. You can see which stage is currently running, overall progress percentage, and estimated time remaining. You can pause or cancel processing at any time.",
      },
    ],
  },
  {
    id: "results-management",
    title: "Results & Scene Management",
    content: "Manage and download your generated 3D scenes.",
    subsections: [
      {
        id: "viewing-results",
        title: "Viewing Results",
        content:
          "After processing completes, view your 3D scene in the interactive viewer. Rotate, zoom, and inspect all extracted objects. View metadata including processing time, file size, and object count.",
      },
      {
        id: "download-formats",
        title: "Download Formats",
        content:
          "Download your scenes in multiple formats:\n- GLB (recommended for web)\n- GLTF (for advanced editing)\n- Metadata JSON (scene information)\nAll formats include textures and materials.",
      },
      {
        id: "my-scenes",
        title: "My Scenes",
        content:
          "Access all your previously generated scenes in the My Scenes section. Search, filter, and organize your scenes. Tag scenes for easy organization and quick access.",
      },
    ],
  },
  {
    id: "profile-settings",
    title: "Profile & Settings",
    content: "Manage your account and preferences.",
    subsections: [
      {
        id: "profile-info",
        title: "Profile Information",
        content:
          "Update your profile with first name, last name, email, and phone number. Keep your information current for account recovery and notifications.",
      },
      {
        id: "security",
        title: "Security",
        content:
          "Change your password anytime from the Security tab. View active sessions across your devices and logout from any device remotely. Enable two-factor authentication for enhanced security.",
      },
      {
        id: "storage",
        title: "Storage & Billing",
        content:
          "Monitor your storage usage and upgrade your plan if needed. View your subscription details and billing history. Manage payment methods and invoices.",
      },
    ],
  },
  {
    id: "troubleshooting",
    title: "Troubleshooting",
    content: "Common issues and solutions.",
    subsections: [
      {
        id: "upload-issues",
        title: "Upload Issues",
        content:
          "If your upload fails:\n- Check file format and size limits\n- Ensure stable internet connection\n- Try uploading a smaller file first\n- Clear browser cache and try again",
      },
      {
        id: "processing-errors",
        title: "Processing Errors",
        content:
          "If processing fails:\n- Check your scene description for clarity\n- Try with a different quality setting\n- Ensure your content is clear and well-lit\n- Contact support if issues persist",
      },
      {
        id: "download-issues",
        title: "Download Issues",
        content:
          "If downloads fail:\n- Check your internet connection\n- Try a different browser\n- Clear browser cache\n- Try downloading in a different format",
      },
    ],
  },
  {
    id: "faq",
    title: "Frequently Asked Questions",
    content: "Quick answers to common questions.",
    subsections: [
      {
        id: "how-long-processing",
        title: "How long does processing take?",
        content:
          "Processing time depends on file size and quality setting. Standard quality typically takes 2-5 minutes, while High Quality can take 5-15 minutes.",
      },
      {
        id: "can-edit-scenes",
        title: "Can I edit my scenes after generation?",
        content:
          "Yes! Download your scene in GLTF format and edit it in 3D software like Blender, Three.js, or Babylon.js. GLB format is optimized for web use.",
      },
      {
        id: "storage-limits",
        title: "What are the storage limits?",
        content:
          "Free plan includes 10GB storage. Premium plans offer 100GB or more. Storage is used for storing your generated scenes and project files.",
      },
      {
        id: "export-options",
        title: "What export options are available?",
        content:
          "Export as GLB (web-optimized), GLTF (editable), or Metadata JSON. All formats include textures, materials, and animations.",
      },
    ],
  },
]

export default function DocsPage() {
  const [expandedSections, setExpandedSections] = useState<string[]>(["getting-started"])
  const [searchQuery, setSearchQuery] = useState("")

  const toggleSection = (id: string) => {
    setExpandedSections((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]))
  }

  const filteredSections = docSections.filter(
    (section) =>
      section.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      section.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      section.subsections?.some(
        (sub) =>
          sub.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          sub.content.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
  )

  return (
    <AppLayout>
      <div className="flex-1 overflow-auto">
        <div className="max-w-4xl mx-auto p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-foreground mb-2">Documentation</h1>
            <p className="text-muted-foreground">Learn how to use SceneForge to create amazing 3D scenes</p>
          </div>

          {/* Search */}
          <div className="mb-8">
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search documentation..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-background border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>

          {/* Documentation Sections */}
          <div className="space-y-4">
            {filteredSections.length > 0 ? (
              filteredSections.map((section) => (
                <div
                  key={section.id}
                  className="bg-card border border-border rounded-lg overflow-hidden hover:border-primary/50 transition-colors"
                >
                  {/* Section Header */}
                  <button
                    onClick={() => toggleSection(section.id)}
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-accent/5 transition-colors"
                  >
                    <div className="text-left">
                      <h2 className="text-xl font-semibold text-foreground">{section.title}</h2>
                      <p className="text-sm text-muted-foreground mt-1">{section.content}</p>
                    </div>
                    <ChevronDown
                      className={`w-5 h-5 text-muted-foreground transition-transform ${
                        expandedSections.includes(section.id) ? "rotate-180" : ""
                      }`}
                    />
                  </button>

                  {/* Section Content */}
                  {expandedSections.includes(section.id) && section.subsections && (
                    <div className="border-t border-border bg-accent/5 px-6 py-4 space-y-4">
                      {section.subsections.map((subsection) => (
                        <div key={subsection.id} className="pb-4 last:pb-0">
                          <h3 className="text-lg font-semibold text-foreground mb-2">{subsection.title}</h3>
                          <p className="text-sm text-muted-foreground whitespace-pre-line">{subsection.content}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No documentation found matching your search.</p>
              </div>
            )}
          </div>

          {/* Support Section */}
          <div className="mt-12 p-6 bg-primary/10 border border-primary/20 rounded-lg">
            <h3 className="text-lg font-semibold text-foreground mb-2">Need Help?</h3>
            <p className="text-muted-foreground mb-4">
              Can't find what you're looking for? Contact our support team for assistance.
            </p>
            <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
              Contact Support
            </button>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
