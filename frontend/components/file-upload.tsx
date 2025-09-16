"use client";

import type React from "react";
import { useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Upload,
  File,
  X,
  CheckCircle,
  AlertTriangle,
  FileText,
  Database,
  Code,
} from "lucide-react";

interface FileUploadProps {
  onFileUpload: (filenames: string[]) => void;
  uploadedFiles: string[];
}

const getFileIcon = (fileType: string) => {
  if (fileType.includes("pdf"))
    return <FileText className="h-4 w-4 text-red-500" />;
  if (fileType.includes("csv"))
    return <Database className="h-4 w-4 text-green-500" />;
  if (fileType.includes("json"))
    return <Code className="h-4 w-4 text-blue-500" />;
  if (fileType.includes("text"))
    return <FileText className="h-4 w-4 text-gray-500" />;
  return <File className="h-4 w-4 text-muted-foreground" />;
};

export function FileUpload({ onFileUpload, uploadedFiles }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const files = Array.from(e.dataTransfer.files);
      const validFiles = files.filter((file) =>
        ["application/pdf", "text/csv", "application/json", "text/plain"].includes(
          file.type
        )
      );

      const rejected = files
        .filter(
          (file) =>
            !["application/pdf", "text/csv", "application/json", "text/plain"].includes(
              file.type
            )
        )
        .map((file) => file.name);

      setRejectedFiles(rejected);

      if (validFiles.length > 0) {
        setUploadStatus("uploading");
        setUploadProgress(0);

        const progressInterval = setInterval(() => {
          setUploadProgress((prev) => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + Math.random() * 15;
          });
        }, 200);

        const uploaded: string[] = [];
        for (const file of validFiles) {
          const filename = await uploadToBackend(file);
          if (filename) uploaded.push(filename);
        }

        clearInterval(progressInterval);
        setUploadProgress(100);
        if (uploaded.length > 0) {
          onFileUpload([...uploadedFiles, ...uploaded]);
        }
        setUploadStatus("success");
        setTimeout(() => setRejectedFiles([]), 5000);
      }
    },
    [onFileUpload, uploadedFiles]
  );

  const uploadToBackend = async (file: File) => {
    setUploadStatus("uploading");
    setErrorMsg(null);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append("file", file);
    try {
  const res = await fetch("http://127.0.0.1:8000/upload-documents/", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setUploadStatus("error");
        setErrorMsg(data.error || "Upload failed");
        return null;
      }
      setUploadStatus("success");
      return data.filename;
    } catch (e) {
      setUploadStatus("error");
      setErrorMsg("Network error");
      return null;
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setUploadStatus("uploading");
      setUploadProgress(0);
      const uploaded: string[] = [];
      for (const file of files) {
        const filename = await uploadToBackend(file);
        if (filename) uploaded.push(filename);
      }
      if (uploaded.length > 0) {
        onFileUpload([...uploadedFiles, ...uploaded]);
      }
    }
  };

  return (
    <Card className="relative overflow-hidden">
      <CardHeader className="relative">
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          File Upload
          {uploadedFiles.length > 0 && (
            <Badge variant="secondary" className="ml-auto">
              {uploadedFiles.length} file{uploadedFiles.length !== 1 ? "s" : ""}
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Upload PDF, CSV, JSON, or TXT files for analysis.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 relative">
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
            isDragOver ? "border-primary bg-primary/10" : "border-border"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploadStatus === "uploading" ? (
            <div className="space-y-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
              <p className="text-sm text-muted-foreground">Uploading...</p>
              <Progress value={uploadProgress} className="w-full max-w-xs mx-auto" />
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="h-12 w-12 text-muted-foreground mx-auto" />
              <p className="font-medium">Drop files here or click to browse</p>
              <input
                type="file"
                multiple
                accept=".pdf,.csv,.json,.txt"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <Button asChild variant="outline">
                <label htmlFor="file-upload" className="cursor-pointer">
                  Choose Files
                </label>
              </Button>
            </div>
          )}
        </div>

        {errorMsg && (
          <div className="text-red-600 text-sm">{errorMsg}</div>
        )}

        {rejectedFiles.length > 0 && (
          <div className="flex items-start gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-destructive">
                Some files were rejected:
              </p>
              <p>{rejectedFiles.join(", ")}</p>
            </div>
          </div>
        )}

        {uploadedFiles.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Uploaded Files:</h4>
            <div className="space-y-2">
              {uploadedFiles.map((filename, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    {/* File icon based on extension */}
                    {getFileIcon(filename)}
                    <p className="text-sm font-medium truncate">{filename}</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      const newFiles = uploadedFiles.filter((_, i) => i !== index);
                      onFileUpload(newFiles);
                    }}
                    className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
