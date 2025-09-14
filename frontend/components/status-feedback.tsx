"use client";
import React from "react";

export type StatusType = "idle" | "loading" | "success" | "error";

export interface StatusFeedbackProps {
  status: StatusType;
  message?: string;
}

const StatusFeedback: React.FC<StatusFeedbackProps> = ({ status, message }) => {
  if (status === "idle") return null;
  if (status === "loading") return (
    <div className="text-blue-600">Loading...</div>
  );
  if (status === "success") return (
    <div className="text-green-600">{message || "Operation successful."}</div>
  );
  if (status === "error") return (
    <div className="text-red-600">{message || "An error occurred."}</div>
  );
  return null;
};

export default StatusFeedback;
