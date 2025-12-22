"use client";

import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { apiUrl } from "@/lib/config";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Eye,
  FileText,
  Database,
  Code,
  Download,
  Loader2,
  AlertCircle,
  FileSpreadsheet,
  File,
} from "lucide-react";

interface FilePreviewProps {
  fileName: string;
  fileType: string;
  isOpen: boolean;
  onClose: () => void;
}

interface PreviewData {
  content?: string;
  data?: any[];
  columns?: string[];
  sheets?: { [key: string]: any };
  metadata?: {
    size: number;
    rows?: number;
    columns?: number;
    encoding?: string;
  };
  error?: string;
}

const getFileIcon = (fileName: string, fileType: string) => {
  if (fileType.includes("pdf") || fileName.endsWith(".pdf"))
    return <FileText className="h-5 w-5 text-red-500" />;
  if (fileType.includes("csv") || fileName.endsWith(".csv"))
    return <Database className="h-5 w-5 text-green-500" />;
  if (fileType.includes("json") || fileName.endsWith(".json"))
    return <Code className="h-5 w-5 text-blue-500" />;
  if (fileName.endsWith(".xlsx") || fileName.endsWith(".xls"))
    return <FileSpreadsheet className="h-5 w-5 text-emerald-500" />;
  if (fileType.includes("text") || fileName.endsWith(".txt"))
    return <FileText className="h-5 w-5 text-gray-500" />;
  return <File className="h-5 w-5 text-muted-foreground" />;
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

export function FilePreview({ fileName, fileType, isOpen, onClose }: FilePreviewProps) {
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("content");

  useEffect(() => {
    if (isOpen && fileName) {
      fetchPreviewData();
    }
  }, [isOpen, fileName]);

  const fetchPreviewData = async () => {
    setIsLoading(true);
    setPreviewData(null);

    try {
      const response = await fetch(
        apiUrl(`api/upload/preview-file/${encodeURIComponent(fileName)}`),
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        setPreviewData({ error: data.error });
      } else {
        setPreviewData(data);
      }
    } catch (error) {
      console.error("Error fetching preview:", error);
      setPreviewData({
        error: `Failed to load preview: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const renderCSVPreview = (data: any[], columns: string[]) => {
    const maxRows = 100; // Limit preview to 100 rows
    const displayData = data.slice(0, maxRows);

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline">{data.length} rows</Badge>
            <Badge variant="outline">{columns.length} columns</Badge>
          </div>
          {data.length > maxRows && (
            <Badge variant="secondary">
              Showing first {maxRows} rows
            </Badge>
          )}
        </div>
        <ScrollArea className="h-[400px] w-full border rounded-md">
          <Table>
            <TableHeader className="sticky top-0 bg-background">
              <TableRow>
                {columns.map((column, index) => (
                  <TableHead key={index} className="whitespace-nowrap">
                    {column}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {displayData.map((row, rowIndex) => (
                <TableRow key={rowIndex}>
                  {columns.map((column, colIndex) => (
                    <TableCell key={colIndex} className="whitespace-nowrap">
                      {String(row[column] || "")}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
      </div>
    );
  };

  const renderExcelPreview = (sheets: { [key: string]: any }) => {
    const sheetNames = Object.keys(sheets);

    if (sheetNames.length === 1) {
      const sheetData = sheets[sheetNames[0]];
      return renderCSVPreview(sheetData.data, sheetData.columns);
    }

    return (
      <Tabs defaultValue={sheetNames[0]} className="w-full">
        <TabsList className="grid w-full grid-cols-auto">
          {sheetNames.map((sheetName) => (
            <TabsTrigger key={sheetName} value={sheetName}>
              {sheetName}
            </TabsTrigger>
          ))}
        </TabsList>
        {sheetNames.map((sheetName) => {
          const sheetData = sheets[sheetName];
          return (
            <TabsContent key={sheetName} value={sheetName}>
              {renderCSVPreview(sheetData.data, sheetData.columns)}
            </TabsContent>
          );
        })}
      </Tabs>
    );
  };

  const renderJSONPreview = (content: string) => {
    try {
      const jsonData = JSON.parse(content);
      const formattedJson = JSON.stringify(jsonData, null, 2);

      return (
        <ScrollArea className="h-[400px] w-full">
          <pre className="text-sm font-mono bg-muted p-4 rounded-md whitespace-pre-wrap">
            {formattedJson}
          </pre>
        </ScrollArea>
      );
    } catch (error) {
      return (
        <ScrollArea className="h-[400px] w-full">
          <pre className="text-sm font-mono bg-muted p-4 rounded-md whitespace-pre-wrap">
            {content}
          </pre>
        </ScrollArea>
      );
    }
  };

  const renderTextPreview = (content: string) => {
    const maxLength = 10000; // Limit preview to 10k characters
    const displayContent = content.length > maxLength
      ? content.substring(0, maxLength) + "\n... (truncated)"
      : content;

    return (
      <div className="space-y-4">
        {content.length > maxLength && (
          <Badge variant="secondary">
            Showing first {maxLength} characters
          </Badge>
        )}
        <ScrollArea className="h-[400px] w-full">
          <pre className="text-sm font-mono bg-muted p-4 rounded-md whitespace-pre-wrap">
            {displayContent}
          </pre>
        </ScrollArea>
      </div>
    );
  };

  const renderMetadata = (metadata: any) => {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">File Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <span className="text-muted-foreground">File Name:</span>
            <span className="font-mono">{fileName}</span>

            {metadata.size && (
              <>
                <span className="text-muted-foreground">Size:</span>
                <span>{formatFileSize(metadata.size)}</span>
              </>
            )}

            {metadata.rows && (
              <>
                <span className="text-muted-foreground">Rows:</span>
                <span>{metadata.rows.toLocaleString()}</span>
              </>
            )}

            {metadata.columns && (
              <>
                <span className="text-muted-foreground">Columns:</span>
                <span>{metadata.columns}</span>
              </>
            )}

            {metadata.encoding && (
              <>
                <span className="text-muted-foreground">Encoding:</span>
                <span>{metadata.encoding}</span>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-[400px]">
          <div className="flex items-center gap-2">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span>Loading preview...</span>
          </div>
        </div>
      );
    }

    if (!previewData) {
      return (
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-center">
            <Eye className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
            <p className="text-muted-foreground">No preview available</p>
          </div>
        </div>
      );
    }

    if (previewData.error) {
      return (
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-destructive mx-auto mb-2" />
            <p className="text-destructive font-medium">Preview Error</p>
            <p className="text-sm text-muted-foreground mt-1">
              {previewData.error}
            </p>
          </div>
        </div>
      );
    }

    // Determine content type and render accordingly
    if (previewData.sheets) {
      return renderExcelPreview(previewData.sheets);
    } else if (previewData.data && previewData.columns) {
      return renderCSVPreview(previewData.data, previewData.columns);
    } else if (previewData.content) {
      if (fileName.endsWith(".json")) {
        return renderJSONPreview(previewData.content);
      } else {
        return renderTextPreview(previewData.content);
      }
    }

    return (
      <div className="flex items-center justify-center h-[400px]">
        <div className="text-center">
          <File className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
          <p className="text-muted-foreground">Unable to preview this file type</p>
        </div>
      </div>
    );
  };

  const availableTabs = [];
  if (previewData && !previewData.error) {
    availableTabs.push("content");
    if (previewData.metadata) {
      availableTabs.push("metadata");
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
        <DialogHeader className="flex-shrink-0">
          <DialogTitle className="flex items-center gap-2">
            {getFileIcon(fileName, fileType)}
            File Preview
          </DialogTitle>
          <DialogDescription className="font-mono text-sm">
            {fileName}
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 min-h-0">
          {availableTabs.length > 1 ? (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
              <TabsList className="flex-shrink-0">
                <TabsTrigger value="content">Content</TabsTrigger>
                {previewData?.metadata && (
                  <TabsTrigger value="metadata">Metadata</TabsTrigger>
                )}
              </TabsList>

              <div className="flex-1 min-h-0">
                <TabsContent value="content" className="h-full">
                  {renderContent()}
                </TabsContent>
                {previewData?.metadata && (
                  <TabsContent value="metadata" className="h-full">
                    {renderMetadata(previewData.metadata)}
                  </TabsContent>
                )}
              </div>
            </Tabs>
          ) : (
            renderContent()
          )}
        </div>

        <div className="flex justify-between items-center pt-4 flex-shrink-0">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button
            variant="outline"
            onClick={() => {
              // Create download link
              const link = document.createElement("a");
              link.href = apiUrl(`api/upload/download-file/${encodeURIComponent(fileName)}`);
              link.download = fileName;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
            }}
          >
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}