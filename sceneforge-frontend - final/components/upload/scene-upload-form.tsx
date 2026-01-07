"use client";

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { api } from '@/lib/api-client';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';

export function SceneUploadForm() {
  const [files, setFiles] = useState<File[]>([]);
  const [sceneName, setSceneName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();
  const router = useRouter();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Check file types and size
    const validFiles = acceptedFiles.filter((file: File) => {
      const isValidType = /^(image\/(jpeg|png|webp)|video\/(mp4|webm))$/i.test((file as File).type);
      const isValidSize = (file as File).size <= 50 * 1024 * 1024; // 50MB limit
      return isValidType && isValidSize;
    });

    if (validFiles.length < acceptedFiles.length) {
      toast({
        title: "Invalid files detected",
        description: "Some files were rejected. Please only upload images (JPG, PNG, WebP) or videos (MP4, WebM) under 50MB.",
        variant: "destructive"
      });
    }

    setFiles(validFiles);
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
      'video/*': ['.mp4', '.webm']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    maxFiles: 50,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Form submission started');
    
    if (!files.length) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload.",
        variant: "destructive"
      });
      return;
    }

    if (!sceneName) {
      toast({
        title: "Scene name required",
        description: "Please provide a name for your scene.",
        variant: "destructive"
      });
      return;
    }

    try {
      setUploading(true);
      setProgress(0);
      console.log('Preparing form data...');

      const formData = new FormData();
      files.forEach((file: File) => {
        console.log(`Adding file: ${file.name} (${file.size} bytes)`);
        formData.append('files', file);
      });
      formData.append('scene_name', sceneName);
      if (prompt) {
        formData.append('prompt', prompt);
      }

      console.log('Submitting to API...');
      // Use API client to upload (routes to backend)
      const data = await api.processing.upload(formData);
      console.log('Upload response:', data);
      
      // Start polling for progress
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.processing.status(data.id);
          setProgress(status.progress);

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            toast({
              title: "Processing completed",
              description: "Your scene has been successfully processed!"
            });
            router.push('/scenes');
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            throw new Error(status.message || 'Processing failed');
          }
        } catch (err: unknown) {
          clearInterval(pollInterval);
          const e = err as Error;
          toast({
            title: 'Error',
            description: e?.message || String(err),
            variant: 'destructive',
          });
        }
      }, 2000);

    } catch (err: unknown) {
      const e = err as Error;
      toast({
        title: 'Error',
        description: e?.message || String(err),
        variant: 'destructive',
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <label className="text-sm font-medium">Scene Name</label>
        <Input
          value={sceneName}
          onChange={(e) => setSceneName(e.target.value)}
          placeholder="Enter a name for your scene"
          required
        />
      </div>

      <div {...getRootProps()} className={`
        border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
        ${isDragActive ? 'border-primary' : 'border-border'}
      `}>
        <input {...getInputProps()} />
        {files.length > 0 ? (
          <div className="space-y-2">
            <p className="text-sm">{files.length} file(s) selected</p>
            <ul className="text-xs text-muted-foreground">
              {files.map((file, index) => (
                <li key={index}>{file.name}</li>
              ))}
            </ul>
          </div>
        ) : isDragActive ? (
          <p>Drop the files here ...</p>
        ) : (
          <div className="space-y-2">
            <p>Drag & drop files here, or click to select files</p>
            <p className="text-xs text-muted-foreground">
              Supported formats: JPG, PNG, WebP (Max 50 images)
            </p>
          </div>
        )}
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Scene Description (Optional)</label>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe how you want your scene to look (e.g., Foggy cyberpunk alley with neon lights...)"
          rows={4}
        />
      </div>

      {uploading && (
        <div className="space-y-2">
          <Progress value={progress} className="w-full" />
          <p className="text-sm text-center text-muted-foreground">
            {progress}% - Processing your scene...
          </p>
        </div>
      )}

      <Button
        type="submit"
        className="w-full"
        disabled={uploading || !files.length}
      >
        {uploading ? 'Processing...' : 'Start Processing'}
      </Button>
    </form>
  );
}