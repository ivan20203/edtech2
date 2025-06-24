'use client';

import { useState } from 'react';

export function PodAgentForm() {
  const [topic, setTopic] = useState('What are the primary factors that influence consumer behavior?');
  const [guestNumber, setGuestNumber] = useState(2);
  const [sessionId, setSessionId] = useState('test');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/PodAgent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic,
          guestNumber,
          sessionId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate podcast');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">PodAgent Podcast Generator</h1>
        <p className="text-gray-600">
          Generate AI-powered podcasts with customizable topics and guest counts
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="topic" className="block text-sm font-medium mb-2">
            Podcast Topic *
          </label>
          <textarea
            id="topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Enter the podcast topic..."
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[100px]"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="guestNumber" className="block text-sm font-medium mb-2">
              Number of Guests
            </label>
            <input
              id="guestNumber"
              type="number"
              min="1"
              max="10"
              value={guestNumber}
              onChange={(e) => setGuestNumber(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label htmlFor="sessionId" className="block text-sm font-medium mb-2">
              Session ID
            </label>
            <input
              id="sessionId"
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="Enter session ID..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !topic.trim()}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Generating Podcast...' : 'Generate Podcast'}
        </button>
      </form>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <h3 className="text-red-800 font-medium">Error</h3>
          <p className="text-red-600 mt-1">{error}</p>
          
          {/* Show suggestions for common errors */}
          {error.includes('Speech index is not in order') && (
            <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded">
              <h4 className="text-yellow-800 font-medium text-sm">💡 Suggestions:</h4>
              <ul className="text-yellow-700 text-sm mt-1 space-y-1">
                <li>• Try a simpler, more specific topic</li>
                <li>• Reduce the number of guests to 1</li>
                <li>• Use a shorter topic description</li>
                <li>• Try a different session ID</li>
              </ul>
            </div>
          )}
          
          {error.includes('Speech item is out of index') && (
            <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded">
              <h4 className="text-yellow-800 font-medium text-sm">💡 Suggestions:</h4>
              <ul className="text-yellow-700 text-sm mt-1 space-y-1">
                <li>• Reduce the number of guests</li>
                <li>• Try a simpler topic</li>
                <li>• Use a different session ID</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="p-4 bg-green-50 border border-green-200 rounded-md">
          <h3 className="text-green-800 font-medium">Success!</h3>
          <div className="text-green-600 mt-2 space-y-1">
            <p><strong>Message:</strong> {result.message}</p>
            <p><strong>Session ID:</strong> {result.session_id}</p>
            <p><strong>Guest Number:</strong> {result.guest_number}</p>
            {result.output && (
              <div>
                <p><strong>Output:</strong></p>
                <pre className="text-sm bg-white p-2 rounded border overflow-auto max-h-40">
                  {result.output}
                </pre>
              </div>
            )}
            
            {/* Audio Files Section */}
            <div className="mt-4">
              <h4 className="font-medium text-green-800">Generated Audio Files:</h4>
              <div className="mt-2 space-y-2">
                <AudioPlayer 
                  sessionId={result.session_id} 
                  fileName="final_podcast.wav" 
                  label="Final Podcast"
                />
                <AudioPlayer 
                  sessionId={result.session_id} 
                  fileName="final_podcast.mp3" 
                  label="Final Podcast (MP3)"
                />
                <AudioPlayer 
                  sessionId={result.session_id} 
                  fileName="mixed_audio.wav" 
                  label="Mixed Audio"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="text-sm text-gray-500 text-center">
        <p>Make sure the PodAgent Flask service is running on port 8021</p>
        <p>Run: <code className="bg-gray-100 px-1 rounded">python services.py</code> in the PodAgent directory</p>
      </div>
    </div>
  );
}

function AudioPlayer({ sessionId, fileName, label }: { sessionId: string; fileName: string; label: string }) {
  const audioUrl = `/api/audio/${sessionId}/${fileName}`;
  
  return (
    <div className="border rounded p-3 bg-white">
      <h5 className="font-medium text-gray-800 mb-2">{label}</h5>
      <audio controls className="w-full">
        <source src={audioUrl} type="audio/wav" />
        <source src={audioUrl.replace('.wav', '.mp3')} type="audio/mpeg" />
        Your browser does not support the audio element.
      </audio>
      <div className="mt-2 flex gap-2">
        <a 
          href={audioUrl} 
          download 
          className="text-sm text-blue-600 hover:text-blue-800 underline"
        >
          Download WAV
        </a>
        <a 
          href={audioUrl.replace('.wav', '.mp3')} 
          download 
          className="text-sm text-blue-600 hover:text-blue-800 underline"
        >
          Download MP3
        </a>
      </div>
    </div>
  );
} 