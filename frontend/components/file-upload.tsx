"use client";
import React, { useRef } from "react";
import { Button } from "./ui/button";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  uploading: boolean;
  fileName: string | null;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, uploading, fileName }) => {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  }

  return (
    <div
      className="border-2 border-dashed border-gray-300 rounded p-4 text-center cursor-pointer bg-gray-50 hover:bg-gray-100 transition flex flex-col items-center gap-2"
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={handleChange}
        accept=".pdf,.csv,.json,.txt"
        title="Upload file"
        placeholder="Select a file"
      />
      {fileName ? (
        <div className="text-green-700 font-medium">{fileName}</div>
      ) : (
        <div>Drag & drop or click to select a file (PDF, CSV, JSON, TXT)</div>
      )}
      <Button
        type="button"
        onClick={e => {
          e.stopPropagation();
          inputRef.current?.click();
        }}
        disabled={uploading}
        className="mt-2"
      >
        {uploading ? "Uploading..." : "Select File"}
      </Button>
    </div>
  );
};

export default FileUpload;
