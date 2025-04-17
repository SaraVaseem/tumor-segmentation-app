"use client"

import type * as React from "react"
import { useCallback, useState } from "react"
// import { FileIcon, UploadCloudIcon, X } from "lucide-react"

import { Button} from "@mui/material"
import { Card, CardContent } from "@mui/material"
import { LinearProgress } from "@mui/material"

interface FileUploadProps {
  onUpload?: (files: File[]) => void
  maxFiles?: number
  maxSize?: number // in bytes
  accept?: string
}

export function FileUpload({
  onUpload,
  maxFiles = 5,
  maxSize = 5 * 1024 * 1024, // 5MB default
  accept = "*",
}: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const [errors, setErrors] = useState<string[]>([])

  const handleFiles = useCallback(
    (selectedFiles: FileList | null) => {
      if (!selectedFiles) return

      setErrors([])

      const newFiles: File[] = []
      const newErrors: string[] = []

      // Convert FileList to array and validate
      Array.from(selectedFiles).forEach((file) => {
        // Check file size
        if (file.size > maxSize) {
          newErrors.push(`"${file.name}" exceeds the maximum file size of ${formatBytes(maxSize)}.`)
          return
        }

        // Check if we've reached the max number of files
        if (files.length + newFiles.length >= maxFiles) {
          newErrors.push(`You can only upload a maximum of ${maxFiles} files.`)
          return
        }

        // Check if file already exists
        if (files.some((f) => f.name === file.name)) {
          newErrors.push(`"${file.name}" has already been added.`)
          return
        }

        newFiles.push(file)
      })

      if (newErrors.length > 0) {
        setErrors(newErrors)
      }

      if (newFiles.length > 0) {
        const updatedFiles = [...files, ...newFiles]
        setFiles(updatedFiles)

        // Initialize progress for new files
        const newProgress = { ...uploadProgress }
        newFiles.forEach((file) => {
          newProgress[file.name] = 0
        })
        setUploadProgress(newProgress)

        // Simulate upload progress
        simulateUpload(newFiles)

        // Call onUpload callback
        if (onUpload) {
          onUpload(updatedFiles)
        }
      }
    },
    [files, maxFiles, maxSize, onUpload, uploadProgress],
  )

  const simulateUpload = (newFiles: File[]) => {
    newFiles.forEach((file) => {
      let progress = 0
      const interval = setInterval(() => {
        progress += Math.floor(Math.random() * 10) + 5
        if (progress >= 100) {
          progress = 100
          clearInterval(interval)
        }

        setUploadProgress((prev) => ({
          ...prev,
          [file.name]: progress,
        }))
      }, 300)
    })
  }

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      handleFiles(e.dataTransfer.files)
    },
    [handleFiles],
  )

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files)
    },
    [handleFiles],
  )

  const removeFile = useCallback((index: number) => {
    setFiles((prevFiles) => {
      const newFiles = [...prevFiles]
      newFiles.splice(index, 1)
      return newFiles
    })
  }, [])

  return (
    <div className="w-full space-y-4">
      <div
        className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors ${
          isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* <UploadCloudIcon className="mb-2 h-10 w-10 text-muted-foreground" /> */}
        <div className="mb-2 text-center">
          <p className="text-sm font-medium">Drag and drop files here or click to browse</p>
          <p className="text-xs text-muted-foreground">
            Upload up to {maxFiles} files (max {formatBytes(maxSize)} each)
          </p>
        </div>
        <input id="file-upload" type="file" multiple accept={accept} onChange={handleInputChange} className="sr-only" />
        <Button onClick={() => document.getElementById("file-upload")?.click()}>
          Select Files
        </Button>
      </div>

      {errors.length > 0 && (
        <div className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
          <ul className="list-inside list-disc space-y-1">
            {errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {files.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <p className="mb-2 text-sm font-medium">Uploaded Files</p>
            <ul className="space-y-2">
              {files.map((file, index) => (
                <li key={index} className="flex items-center justify-between rounded-md border p-2">
                  <div className="flex items-center space-x-2">
                    {/* <FileIcon className="h-5 w-5 text-muted-foreground" /> */}
                    <div>
                      <p className="text-sm font-medium">{file.name}</p>
                      <p className="text-xs text-muted-foreground">{formatBytes(file.size)}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {uploadProgress[file.name] < 100 ? (
                      <div className="w-24">
                        <LinearProgress value={uploadProgress[file.name]} className="h-2" />
                      </div>
                    ) : (
                      <span className="text-xs text-green-600">Complete</span>
                    )}
                    <Button className="h-7 w-7" onClick={() => removeFile(index)}>
                      {/* <X className="h-4 w-4" /> */}
                      <span className="sr-only">Remove file</span>
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function formatBytes(bytes: number, decimals = 2) {
  if (bytes === 0) return "0 Bytes"

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"]

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i]
}
