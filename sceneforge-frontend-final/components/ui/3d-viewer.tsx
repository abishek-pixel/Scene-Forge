"use client"

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'

interface Viewer3DProps {
  modelUrl?: string
  loading?: boolean
  error?: string
  onLoaded?: () => void
}

export function Viewer3D({ modelUrl, loading = false, error, onLoaded }: Viewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<any>(null)
  const meshRef = useRef<THREE.Object3D | null>(null)
  const [isInitialized, setIsInitialized] = useState(false)

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current || isInitialized) return

    // Create scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1a)
    sceneRef.current = scene

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      10000
    )
    camera.position.set(0, 0, 5)
    cameraRef.current = camera

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    containerRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(5, 10, 5)
    directionalLight.castShadow = true
    directionalLight.shadow.mapSize.width = 2048
    directionalLight.shadow.mapSize.height = 2048
    directionalLight.shadow.camera.far = 50
    scene.add(directionalLight)

    const pointLight = new THREE.PointLight(0xffffff, 0.4)
    pointLight.position.set(-5, 5, 5)
    scene.add(pointLight)

    // Add basic grid
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222)
    scene.add(gridHelper)

    // Add axes helper
    const axesHelper = new THREE.AxesHelper(5)
    scene.add(axesHelper)

    // Simple orbit controls replacement (mouse wheel zoom, click drag rotate)
    let isDragging = false
    let previousMousePosition = { x: 0, y: 0 }

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true
      previousMousePosition = { x: e.clientX, y: e.clientY }
    }

    const onMouseMove = (e: MouseEvent) => {
      if (isDragging && meshRef.current) {
        const deltaX = e.clientX - previousMousePosition.x
        const deltaY = e.clientY - previousMousePosition.y

        meshRef.current.rotation.y += deltaX * 0.01
        meshRef.current.rotation.x += deltaY * 0.01

        previousMousePosition = { x: e.clientX, y: e.clientY }
      }
    }

    const onMouseUp = () => {
      isDragging = false
    }

    const onMouseWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (cameraRef.current) {
        const zoomSpeed = 0.1
        const direction = e.deltaY > 0 ? 1 : -1
        cameraRef.current.position.z += direction * zoomSpeed
      }
    }

    renderer.domElement.addEventListener('mousedown', onMouseDown)
    renderer.domElement.addEventListener('mousemove', onMouseMove)
    renderer.domElement.addEventListener('mouseup', onMouseUp)
    renderer.domElement.addEventListener('wheel', onMouseWheel, { passive: false })

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return
      const width = containerRef.current.clientWidth
      const height = containerRef.current.clientHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }

    window.addEventListener('resize', handleResize)

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate)
      renderer.render(scene, camera)
    }

    animate()

    setIsInitialized(true)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      renderer.domElement.removeEventListener('mousedown', onMouseDown)
      renderer.domElement.removeEventListener('mousemove', onMouseMove)
      renderer.domElement.removeEventListener('mouseup', onMouseUp)
      renderer.domElement.removeEventListener('wheel', onMouseWheel)
    }
  }, [isInitialized])

  // Load model when URL changes
  useEffect(() => {
    if (!modelUrl || !sceneRef.current || !cameraRef.current) return

    const loader = new GLTFLoader()

    loader.load(
      modelUrl,
      (gltf: any) => {
        console.log('Model loaded successfully:', gltf)
        // Remove old mesh if exists
        if (meshRef.current) {
          sceneRef.current?.remove(meshRef.current)
        }

        const model = gltf.scene
        meshRef.current = model

        // Calculate bounds and center
        const box = new THREE.Box3().setFromObject(model)
        const center = box.getCenter(new THREE.Vector3())
        const size = box.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z)
        const fov = cameraRef.current!.fov * (Math.PI / 180)
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2))
        cameraZ *= 1.5

        model.position.sub(center)
        cameraRef.current!.position.z = cameraZ
        cameraRef.current!.lookAt(0, 0, 0)

        sceneRef.current!.add(model)

        // Enable shadows
        model.traverse((child: any) => {
          if (child.isMesh) {
            child.castShadow = true
            child.receiveShadow = true
          }
        })

        console.log('Model added to scene and shadows enabled')
        onLoaded?.()
      },
      (progress: any) => {
        const percent = (progress.loaded / progress.total * 100).toFixed(2)
        console.log(`Loading model: ${percent}%`)
      },
      (error: any) => {
        console.error('Failed to load model:', error)
        console.error('Model URL:', modelUrl)
        console.error('Error details:', {
          message: error.message,
          stack: error.stack
        })
      }
    )
  }, [modelUrl, onLoaded])

  return (
    <div
      ref={containerRef}
      className="w-full h-full bg-slate-900 rounded-lg overflow-hidden relative"
      style={{ minHeight: '400px' }}
    >
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto mb-4" />
            <p className="text-white">Loading 3D model...</p>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-center">
            <p className="text-red-400 font-semibold">{error}</p>
          </div>
        </div>
      )}
      {!modelUrl && !loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl text-primary/30 mb-4">3D</div>
            <p className="text-muted-foreground">No model loaded</p>
            <p className="text-xs text-muted-foreground mt-2">Waiting for model URL...</p>
          </div>
        </div>
      )}
    </div>
  )
}
