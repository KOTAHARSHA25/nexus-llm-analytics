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
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiUrl, getEndpoint } from "@/lib/config";
import {
  Upload,
  File,
  X,
  CheckCircle,
  AlertTriangle,
  FileText,
  Database,
  Code,
  Eye,
  Download,
  ClipboardPaste,
} from "lucide-react";
import { FilePreview } from "./file-preview";
import { FileInfo } from "@/hooks/useDashboardState";

interface FileUploadProps {
  onFileUpload: (files: FileInfo[]) => void;
  uploadedFiles: FileInfo[];
}

const getFileIcon = (fileType: string) => {
  if (fileType.includes("pdf"))
    return <FileText className="h-4 w-4 text-red-400" />;
  if (fileType.includes("csv"))
    return <Database className="h-4 w-4 text-green-400" />;
  if (fileType.includes("json"))
    return <Code className="h-4 w-4 text-blue-400" />;
  if (fileType.includes("text"))
    return <FileText className="h-4 w-4 text-purple-400" />;
  return <File className="h-4 w-4 text-gray-400" />;
};

export function FileUpload({ onFileUpload, uploadedFiles }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<{
    fileName: string;
    fileType: string;
  } | null>(null);
  
  // Paste text feature state
  const [pasteTitle, setPasteTitle] = useState("");
  const [pasteText, setPasteText] = useState("");
  const [pasteDescription, setPasteDescription] = useState("");
  const [isPastingText, setIsPastingText] = useState(false);

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

        const uploaded: FileInfo[] = [];
        for (const file of validFiles) {
          const fileInfo = await uploadToBackend(file);
          if (fileInfo) uploaded.push(fileInfo);
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

  const uploadToBackend = async (file: File): Promise<FileInfo | null> => {
    setUploadStatus("uploading");
    setErrorMsg(null);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(getEndpoint('uploadDocuments'), {
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

      // Return file info with metadata
      return {
        name: data.filename || file.name,
        type: file.type,
        columns: data.columns || [],
        id: crypto.randomUUID(),
        uploadedAt: Date.now()
      };
    } catch (e) {
      setUploadStatus("error");
      setErrorMsg("Network error");
      return null;
    }
  };

  const handleDownloadFile = async (filename: string) => {
    try {
      const response = await fetch(`${getEndpoint('downloadFile')}/${encodeURIComponent(filename)}`);

      if (!response.ok) {
        throw new Error('Download failed');
      }

      // Get the blob data
      const blob = await response.blob();

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();

      // Cleanup
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      setErrorMsg(`Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setUploadStatus("error");
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setUploadStatus("uploading");
      setUploadProgress(0);
      const uploaded: FileInfo[] = [];
      for (const file of files) {
        const fileInfo = await uploadToBackend(file);
        if (fileInfo) uploaded.push(fileInfo);
      }
      if (uploaded.length > 0) {
        onFileUpload([...uploadedFiles, ...uploaded]);
      }
    }
  };

  const handlePasteTextSubmit = async () => {
    if (!pasteText.trim()) {
      setErrorMsg("Please enter some text to upload");
      return;
    }
    
    setIsPastingText(true);
    setErrorMsg(null);
    
    try {
      const response = await fetch(getEndpoint('uploadRawText'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: pasteTitle || 'Pasted Text',
          text: pasteText,
          description: pasteDescription || 'Text pasted directly'
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok || data.error) {
        setErrorMsg(data.error || 'Failed to upload text');
        setIsPastingText(false);
        return;
      }
      
      // Add to uploaded files list
      const newFileInfo: FileInfo = {
        name: data.filename,
        type: 'text/plain',
        columns: [],
        id: crypto.randomUUID(),
        uploadedAt: Date.now()
      };
      
      onFileUpload([...uploadedFiles, newFileInfo]);
      
      // Clear form
      setPasteTitle("");
      setPasteText("");
      setPasteDescription("");
      setUploadStatus("success");
      
    } catch (error) {
      setErrorMsg(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsPastingText(false);
    }
  };

  return (
    <Card className="glass-card hover:glow-card transition-all duration-300">
      <CardHeader className="relative">
        <CardTitle className="flex items-center gap-2 text-white">
          <Upload className="h-5 w-5 text-purple-300" />
          Data Input
          {uploadedFiles.length > 0 && (
            <Badge className="ml-auto bg-gradient-to-r from-purple-500 to-blue-500 text-white border-0">
              {uploadedFiles.length} file{uploadedFiles.length !== 1 ? "s" : ""}
            </Badge>
          )}
        </CardTitle>
        <CardDescription className="text-gray-300">
          Upload files or paste text directly for analysis.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 relative">
        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-gray-800/50">
            <TabsTrigger value="upload" className="data-[state=active]:bg-purple-500/30">
              <Upload className="h-4 w-4 mr-2" />
              Upload File
            </TabsTrigger>
            <TabsTrigger value="paste" className="data-[state=active]:bg-purple-500/30">
              <ClipboardPaste className="h-4 w-4 mr-2" />
              Paste Text
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="upload" className="mt-4">
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${isDragOver
                  ? "border-purple-400 bg-gradient-to-br from-purple-500/20 to-blue-500/20 shadow-lg shadow-purple-500/25"
                  : "border-purple-300/50 hover:border-purple-400/70 hover:bg-gradient-to-br hover:from-purple-500/10 hover:to-blue-500/10"
                }`}
              onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploadStatus === "uploading" ? (
            <div className="space-y-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto"></div>
              <p className="text-sm text-gray-300">Uploading...</p>
              <Progress value={uploadProgress} className="w-full max-w-xs mx-auto bg-gray-700" />
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="h-12 w-12 text-purple-300 mx-auto" />
              <p className="font-medium text-white">Drop files here or click to browse</p>
              <p className="text-xs text-gray-400">Supports PDF, CSV, JSON, TXT, XLSX</p>
              <input
                type="file"
                multiple
                accept=".pdf,.csv,.json,.txt,.xlsx,.xls"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <Button
                asChild
                className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white border-0 shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
              >
                <label htmlFor="file-upload" className="cursor-pointer">
                  Choose Files
                </label>
              </Button>
            </div>
          )}
            </div>
          </TabsContent>
          
          <TabsContent value="paste" className="mt-4 space-y-4">
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-300 mb-1 block">Title</label>
                <Input
                  placeholder="e.g., Sales Data Q4 2025"
                  value={pasteTitle}
                  onChange={(e) => setPasteTitle(e.target.value)}
                  className="bg-gray-800/50 border-gray-600 text-white placeholder:text-gray-500"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-gray-300 mb-1 block">Text Content</label>
                <Textarea
                  placeholder="Paste your data here... (CSV, JSON, SQL, or any text)"
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  className="bg-gray-800/50 border-gray-600 text-white placeholder:text-gray-500 min-h-[150px] font-mono text-sm"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-gray-300 mb-1 block">Description (optional)</label>
                <Input
                  placeholder="Brief description of the data"
                  value={pasteDescription}
                  onChange={(e) => setPasteDescription(e.target.value)}
                  className="bg-gray-800/50 border-gray-600 text-white placeholder:text-gray-500"
                />
              </div>
              <Button
                onClick={handlePasteTextSubmit}
                disabled={isPastingText || !pasteText.trim()}
                className="w-full bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white border-0 shadow-lg hover:shadow-purple-500/25 transition-all duration-300"
              >
                {isPastingText ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Uploading...
                  </>
                ) : (
                  <>
                    <ClipboardPaste className="h-4 w-4 mr-2" />
                    Upload Text
                  </>
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        {errorMsg && (
          <div className="text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">{errorMsg}</div>
        )}

        {rejectedFiles.length > 0 && (
          <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-red-400 mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-red-400">
                Some files were rejected:
              </p>
              <p className="text-gray-300">{rejectedFiles.join(", ")}</p>
            </div>
          </div>
        )}

        {uploadedFiles.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-white">Uploaded Files:</h4>
            <div className="space-y-2">
              {uploadedFiles.map((fileInfo, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-800/50 to-gray-700/50 rounded-lg border border-gray-600/50 hover:border-purple-400/50 transition-all duration-300"
                >
                  <div className="flex items-center gap-3 flex-1">
                    {/* File icon based on type */}
                    {getFileIcon(fileInfo.type || fileInfo.name)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate text-white">{fileInfo.name}</p>
                      {fileInfo.columns && fileInfo.columns.length > 0 && (
                        <p className="text-xs text-gray-400 truncate">
                          Columns: {fileInfo.columns.slice(0, 3).join(', ')}
                          {fileInfo.columns.length > 3 && ` +${fileInfo.columns.length - 3} more`}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setPreviewFile({
                          fileName: fileInfo.name,
                          fileType: fileInfo.type || ""
                        });
                      }}
                      className="h-8 w-8 hover:bg-purple-500/20 hover:text-purple-300 text-gray-400 transition-all duration-300"
                      title="Preview file"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDownloadFile(fileInfo.name)}
                      className="h-8 w-8 hover:bg-green-500/20 hover:text-green-300 text-gray-400 transition-all duration-300"
                      title="Download file"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        const newFiles = uploadedFiles.filter((_, i) => i !== index);
                        onFileUpload(newFiles);
                      }}
                      className="h-8 w-8 hover:bg-red-500/20 hover:text-red-400 text-gray-400 transition-all duration-300"
                      title="Remove file"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* File Preview Modal */}
        {previewFile && (
          <FilePreview
            fileName={previewFile.fileName}
            fileType={previewFile.fileType}
            isOpen={!!previewFile}
            onClose={() => setPreviewFile(null)}
          />
        )}
      </CardContent>
    </Card>
  );
}
