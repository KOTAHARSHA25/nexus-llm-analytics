import React from 'react';

export type StatusType = 'idle' | 'loading' | 'success' | 'error';

export interface StatusFeedbackProps {
  status: StatusType;
  message?: string;
}

export const StatusFeedback: React.FC<StatusFeedbackProps> = ({ status, message }) => {
  if (status === 'idle') return null;
  if (status === 'loading') return (
    <div className="status-feedback loading">Loading...</div>
  );
  if (status === 'success') return (
    <div className="status-feedback success">{message || 'Operation successful.'}</div>
  );
  if (status === 'error') return (
    <div className="status-feedback error">{message || 'An error occurred.'}</div>
  );
  return null;
};
