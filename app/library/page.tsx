'use client';

import { useState, useEffect } from 'react';
import { Download, FileAudio, Play, Loader2 } from 'lucide-react';

interface AudioFile {
  id: string;
  name: string;
  url: string;
  size?: string;
  date?: string;
}

// List of .wav files from RunPod
const audioFiles: AudioFile[] = [
  {
    id: '1',
    name: 'podcast_1753759687.wav',
    url: '/api/MoonCast/download/podcast_1753759687.wav',
    size: '36.8 MB',
    date: '2024-01-29'
  }
];

export default function LibraryPage() {
  const [downloading, setDownloading] = useState<string | null>(null);
  const [files, setFiles] = useState<AudioFile[]>([]);
  const [generating, setGenerating] = useState(false);
  const [topic, setTopic] = useState('');
  const [duration, setDuration] = useState(1);

  // Load files on component mount
  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const response = await fetch('/api/MoonCast/files');
      const data = await response.json();
      setFiles(data.files.map((file: any, index: number) => ({
        id: index.toString(),
        name: file.name,
        url: `/api/MoonCast/download/${file.name}`,
        size: `${(file.size / 1024 / 1024).toFixed(1)} MB`,
        date: new Date(file.created * 1000).toLocaleDateString()
      })));
    } catch (error) {
      console.error('Failed to load files:', error);
    }
  };

  const handleGenerate = async () => {
    if (!topic.trim()) {
      alert('Please enter a topic');
      return;
    }

    setGenerating(true);
    try {
      // Start generation
      const startResponse = await fetch('/api/MoonCast/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, duration })
      });

      if (!startResponse.ok) {
        throw new Error('Failed to start generation');
      }

      const { job_id } = await startResponse.json();

      // Poll for completion
      while (true) {
        const statusResponse = await fetch(`/api/MoonCast/status/${job_id}`);
        const status = await statusResponse.json();

        if (status.status === 'completed') {
          alert('Podcast generated successfully!');
          loadFiles(); // Reload the file list
          break;
        } else if (status.status === 'failed') {
          throw new Error(status.error || 'Generation failed');
        }

        // Wait 2 seconds before checking again
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    } catch (error) {
      console.error('Generation failed:', error);
      alert('Generation failed: ' + error);
    } finally {
      setGenerating(false);
    }
  };

  const handleDownload = async (file: AudioFile) => {
    setDownloading(file.id);
    try {
      const response = await fetch(file.url);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = file.name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Generated Podcasts
          </h1>
          <p className="text-slate-600 dark:text-slate-300">
            Download your AI-generated podcast files
          </p>
          <div className="mt-4 text-sm text-slate-500 dark:text-slate-400">
            {files.length} file{files.length !== 1 ? 's' : ''} available
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
          <div className="space-y-4">
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-4 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <FileAudio className="w-6 h-6 text-blue-500" />
                  <div>
                    <h3 className="font-medium text-slate-900 dark:text-white">
                      {file.name}
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      {file.size} • {file.date}
                    </p>
                    <p className="text-xs text-slate-400 dark:text-slate-500">
                      Duration: ~{Math.round(parseFloat(file.size || '0') * 0.3)} minutes
                    </p>
                  </div>
                </div>
                
                <button
                  onClick={() => handleDownload(file)}
                  disabled={downloading === file.id}
                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-400 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>
                    {downloading === file.id ? 'Downloading...' : 'Download'}
                  </span>
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
