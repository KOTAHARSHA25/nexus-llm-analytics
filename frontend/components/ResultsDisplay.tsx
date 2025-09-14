import React from 'react';
import { StatusFeedback, StatusType } from './StatusFeedback';

interface ResultsDisplayProps {
	status: StatusType;
	message?: string;
	results?: any;
}

// Displays results: text, tables, and charts, with error/status feedback
export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ status, message, results }) => {
	return (
		<div>
			<StatusFeedback status={status} message={message} />
			{/* Render results here (table, chart, etc.) */}
			{results && <pre>{JSON.stringify(results, null, 2)}</pre>}
		</div>
	);
};
