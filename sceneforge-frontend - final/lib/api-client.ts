"use client";

import { toast } from "sonner";

// Ensure we always use http://localhost:8000 in development
const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
const API_URL = rawApiUrl.replace(/\/$/, '');

console.log('API URL configured as:', API_URL); // Debug log

interface FetchOptions extends RequestInit {
  token?: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse(response: Response) {
  const data = await response.json();
  
  if (!response.ok) {
    const error = (data && data.detail) || response.statusText;
    throw new ApiError(response.status, error);
  }
  
  return data;
}

export async function fetchApi(endpoint: string, options: FetchOptions = {}) {
  const { token, ...fetchOptions } = options;
  
  const headers = new Headers(options.headers || {});
  
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  
  headers.set('Content-Type', 'application/json');
  
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      ...fetchOptions,
      headers,
    });
    
    return await handleResponse(response);
  } catch (error) {
    if (error instanceof ApiError) {
      toast.error(error.message);
      throw error;
    }
    toast.error('Network error occurred');
    throw new Error('Network error occurred');
  }
}

// Upload form data helper (don't set JSON content-type)
export async function uploadForm(endpoint: string, formData: FormData, token?: string) {
  try {
    // Ensure proper URL construction with slash handling
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${API_URL}${cleanEndpoint}`;
    console.log('Starting upload to:', url);

    // Log form data contents
    console.log('Form data contents:');
    for (let [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(`${key}: File - ${value.name} (${value.size} bytes)`);
      } else {
        console.log(`${key}: ${value}`);
      }
    }

    // Log the upload attempt
    const uploadStartTime = new Date();
    console.log(`Starting upload at ${uploadStartTime.toISOString()}`);

    const headers: Record<string, string> = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    console.log('Sending upload request...');
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers,
      mode: 'cors', // Explicitly set CORS mode
      credentials: 'include',
    });
    
    const uploadEndTime = new Date();
    const uploadDuration = (uploadEndTime.getTime() - uploadStartTime.getTime()) / 1000;
    console.log(`Upload completed in ${uploadDuration.toFixed(2)} seconds`);

    console.log('Upload response status:', response.status);
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload error:', errorText);
      throw new ApiError(response.status, errorText);
    }

    const responseData = await response.json();
    console.log('Upload response data:', responseData);
    return responseData;
  } catch (error) {
    console.error('Upload error:', error);
    if (error instanceof ApiError) {
      toast.error(error.message);
      throw error;
    }
    toast.error('Network error occurred');
    throw new Error('Network error occurred');
  }
}
// API utilities for different endpoints
export const api = {
  auth: {
    login: (credentials: { email: string; password: string }) =>
      fetchApi('/auth/login', {
        method: 'POST',
        body: JSON.stringify(credentials),
      }),
    register: (userData: { email: string; password: string; name: string }) =>
      fetchApi('/auth/register', {
        method: 'POST',
        body: JSON.stringify(userData),
      }),
    verifyEmail: (token: string) =>
      fetchApi('/auth/verify-email', {
        method: 'POST',
        body: JSON.stringify({ token }),
      }),
    profile: (token: string) =>
      fetchApi('/auth/profile', {
        token,
      }),
  },
  scenes: {
    list: (token: string) =>
      fetchApi('/scenes', {
        token,
      }),
    create: (token: string, sceneData: any) =>
      fetchApi('/scenes', {
        method: 'POST',
        body: JSON.stringify(sceneData),
        token,
      }),
    get: (token: string, sceneId: string) =>
      fetchApi(`/scenes/${sceneId}`, {
        token,
      }),
    update: (token: string, sceneId: string, sceneData: any) =>
      fetchApi(`/scenes/${sceneId}`, {
        method: 'PUT',
        body: JSON.stringify(sceneData),
        token,
      }),
    delete: (token: string, sceneId: string) =>
      fetchApi(`/scenes/${sceneId}`, {
        method: 'DELETE',
        token,
      }),
  },
  processing: {
    start: (token: string, data: any) =>
      fetchApi('/processing/start', {
        method: 'POST',
        body: JSON.stringify(data),
        token,
      }),
    // Upload files using FormData. Usage: api.processing.upload(formData)
    upload: (formData: FormData, token?: string) =>
      uploadForm('/processing/files', formData, token),  // Keep the leading slash
    // Get processing status by job id
    status: (jobId: string, token?: string) =>
      fetchApi(`/processing/job/${jobId}`, {  // Keep the leading slash
        token,
      }),
    // List all processing tasks
    list: (token?: string) =>
      fetchApi('/processing/tasks', {
        token,
      }),
    // List completed scenes
    scenes: (token?: string) =>
      fetchApi('/processing/scenes', {
        token,
      }),
    // Pause a processing task
    pause: (jobId: string, token?: string) =>
      fetchApi(`/processing/${jobId}/pause`, {
        method: 'POST',
        token,
      }),
    // Cancel a processing task
    cancel: (jobId: string, token?: string) =>
      fetchApi(`/processing/${jobId}/cancel`, {
        method: 'POST',
        token,
      }),
  },
};