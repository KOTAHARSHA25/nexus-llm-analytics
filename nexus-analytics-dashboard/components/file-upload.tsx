"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Upload, File, X, CheckCircle, AlertTriangle, FileText, Database, Code } from "lucide-react"

interface FileUploadProps {
  onFileUpload: (files: File[]) => void
  uploadedFiles: File[]
}

const getFileIcon = (fileType: string) => {
  if (fileType.includes("pdf")) return <FileText className="h-4 w-4 text-red-500" />
  if (fileType.includes("csv")) return <Database className="h-4 w-4 text-green-500" />
  if (fileType.includes("json")) return <Code className="h-4 w-4 text-blue-500" />
  if (fileType.includes("text")) return <FileText className="h-4 w-4 text-gray-500" />
  return <File className="h-4 w-4 text-muted-foreground" />
}

export function FileUpload({ onFileUpload, uploadedFiles }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success">("idle")
  const [uploadProgress, setUploadProgress] = useState(0)
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)

      const files = Array.from(e.dataTransfer.files)
      const validFiles = files.filter((file) =>
        ["application/pdf", "text/csv", "application/json", "text/plain"].includes(file.type),
      )

      const rejected = files
        .filter((file) => !["application/pdf", "text/csv", "application/json", "text/plain"].includes(file.type))
        .map((file) => file.name)

      setRejectedFiles(rejected)

      if (validFiles.length > 0) {
        setUploadStatus("uploading")
        setUploadProgress(0)

        const progressInterval = setInterval(() => {
          setUploadProgress((prev) => {
            if (prev >= 90) {
              clearInterval(progressInterval)
              return 90
            }
            return prev + Math.random() * 15
          })
        }, 200)

        setTimeout(() => {
          clearInterval(progressInterval)
          setUploadProgress(100)
          onFileUpload(validFiles)
          setUploadStatus("success")
          setTimeout(() => setRejectedFiles([]), 5000)
        }, 1500)
      }
    },
    [onFileUpload],
  )

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      setUploadStatus("uploading")
      setUploadProgress(0)

      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + Math.random() * 15
        })
      }, 200)

      setTimeout(() => {
        clearInterval(progressInterval)
        setUploadProgress(100)
        onFileUpload(files)
        setUploadStatus("success")
      }, 1500)
    }
  }

  const removeFile = (index: number) => {
    const newFiles = uploadedFiles.filter((_, i) => i !== index)
    onFileUpload(newFiles)
    if (newFiles.length === 0) {
      setUploadStatus("idle")
      setUploadProgress(0)
    }
  }

  return (
    <Card className="relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-transparent pointer-events-none" />

      <CardHeader className="relative">
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5 text-accent" />
          File Upload
          {uploadedFiles.length > 0 && (
            <Badge variant="secondary" className="ml-auto">
              {uploadedFiles.length} file{uploadedFiles.length !== 1 ? "s" : ""}
            </Badge>
          )}
        </CardTitle>
        <CardDescription>Upload PDF, CSV, JSON, or TXT files for analysis (Max 10MB per file)</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 relative">
        {/* Drop zone */}
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
            isDragOver
              ? "border-accent bg-accent/10 scale-[1.02] shadow-lg"
              : uploadStatus === "success"
                ? "border-green-500 bg-green-50 dark:bg-green-950/20"
                : "border-border hover:border-accent/50 hover:bg-accent/5"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploadStatus === "uploading" ? (
            <div className="space-y-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent mx-auto"></div>
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Uploading files...</p>
                <Progress value={uploadProgress} className="w-full max-w-xs mx-auto" />
                <p className="text-xs text-muted-foreground">{Math.round(uploadProgress)}% complete</p>
              </div>
            </div>
          ) : uploadStatus === "success" && uploadedFiles.length > 0 ? (
            <div className="space-y-2">
              <CheckCircle className="h-8 w-8 text-green-500 mx-auto" />
              <p className="text-sm font-medium">Files uploaded successfully!</p>
              <p className="text-xs text-muted-foreground">Ready for analysis</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <Upload className="h-12 w-12 text-muted-foreground mx-auto" />
                {isDragOver && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-16 h-16 border-2 border-accent border-dashed rounded-full animate-pulse" />
                  </div>
                )}
              </div>
              <div>
                <p className="text-lg font-medium">Drop files here or click to browse</p>
                <p className="text-sm text-muted-foreground">Supports PDF, CSV, JSON, and TXT files</p>
              </div>
              <input
                type="file"
                multiple
                accept=".pdf,.csv,.json,.txt"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <Button asChild variant="outline" className="hover:bg-accent hover:text-accent-foreground bg-transparent">
                <label htmlFor="file-upload" className="cursor-pointer">
                  Choose Files
                </label>
              </Button>
            </div>
          )}
        </div>

        {rejectedFiles.length > 0 && (
          <div className="flex items-start gap-2 p-3 bg-orange-50 dark:bg-orange-950/20 border border-orange-200 dark:border-orange-800 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-orange-800 dark:text-orange-200">Some files were rejected:</p>
              <p className="text-orange-700 dark:text-orange-300">{rejectedFiles.join(", ")}</p>
              <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                Only PDF, CSV, JSON, and TXT files are supported.
              </p>
            </div>
          </div>
        )}

        {/* Enhanced uploaded files list */}
        {uploadedFiles.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Uploaded Files:</h4>
              <Badge variant="outline" className="text-xs">
                {uploadedFiles.reduce((acc, file) => acc + file.size, 0) > 1024 * 1024
                  ? `${(uploadedFiles.reduce((acc, file) => acc + file.size, 0) / 1024 / 1024).toFixed(1)} MB total`
                  : `${Math.round(uploadedFiles.reduce((acc, file) => acc + file.size, 0) / 1024)} KB total`}
              </Badge>
            </div>
            <div className="space-y-2">
              {uploadedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border hover:bg-muted/70 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {getFileIcon(file.type)}
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium truncate">{file.name}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                        <span>•</span>
                        <span>{new Date(file.lastModified).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {file.type.split("/")[1].toUpperCase()}
                    </Badge>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => removeFile(index)}
                      className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
