'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX, Download, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

interface AudioTrack {
  id: string;
  title: string;
  artist: string;
  duration: string;
  file: string;
  waveform: number[];
  fileType: string;
}

// Dynamic tracks based on actual audio files in public/audio folder
const audioTracks: AudioTrack[] = [
  {
    id: '1',
    title: 'DIA',
    artist: 'Audio Track',
    duration: '0:00', // Will be updated when loaded
    file: '/audio/DIA.wav',
    fileType: 'wav',
    waveform: [0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.6, 0.8, 0.9, 0.7, 0.4, 0.2, 0.5, 0.7]
  },
  {
    id: '2',
    title: 'DIA2',
    artist: 'Audio Track',
    duration: '0:00',
    file: '/audio/DIA2.wav',
    fileType: 'wav',
    waveform: [0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.5, 0.8, 0.9, 0.6, 0.3, 0.7, 0.9, 0.5]
  },
  {
    id: '3',
    title: 'DIA24sec',
    artist: 'Audio Track',
    duration: '0:00',
    file: '/audio/DIA24sec.wav',
    fileType: 'wav',
    waveform: [0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.6, 0.8, 0.9, 0.7, 0.4, 0.2, 0.5, 0.7, 0.9]
  },
  {
    id: '4',
    title: 'Mooncast',
    artist: 'Podcast Audio',
    duration: '0:00',
    file: '/audio/Mooncast.mp3',
    fileType: 'mp3',
    waveform: [0.1, 0.3, 0.6, 0.8, 0.9, 0.7, 0.4, 0.2, 0.6, 0.9, 0.8, 0.5, 0.3, 0.7, 0.9, 0.6]
  },
  {
    id: '5',
    title: 'Mooncast2',
    artist: 'Podcast Audio',
    duration: '0:00',
    file: '/audio/Mooncast2.mp3',
    fileType: 'mp3',
    waveform: [0.2, 0.4, 0.6, 0.7, 0.8, 0.6, 0.4, 0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.2, 0.5, 0.7]
  },
  {
    id: '6',
    title: 'MoonCast39sec',
    artist: 'Podcast Audio',
    duration: '0:00',
    file: '/audio/MoonCast39sec.mp3',
    fileType: 'mp3',
    waveform: [0.3, 0.5, 0.7, 0.8, 0.9, 0.6, 0.4, 0.2, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.7, 0.9]
  },
  {
    id: '7',
    title: 'PodAgent',
    artist: 'Podcast Agent',
    duration: '0:00',
    file: '/audio/PodAgent.wav',
    fileType: 'wav',
    waveform: [0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.6, 0.8, 0.9, 0.7, 0.4, 0.2, 0.5, 0.7, 0.9]
  },
  {
    id: '8',
    title: 'PodAgent2',
    artist: 'Podcast Agent',
    duration: '0:00',
    file: '/audio/PodAgent2.wav',
    fileType: 'wav',
    waveform: [0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 0.8, 0.6, 0.3, 0.7, 0.9, 0.5, 0.8]
  }
];

export default function DemoPage() {
  const [currentTrack, setCurrentTrack] = useState<AudioTrack | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);
  const [showVolumeSlider, setShowVolumeSlider] = useState(false);
  const [tracks, setTracks] = useState<AudioTrack[]>(audioTracks);
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const volumeSliderRef = useRef<HTMLDivElement>(null);

  // Function to update track duration when audio metadata is loaded
  const updateTrackDuration = (trackId: string, duration: number) => {
    setTracks(prevTracks => 
      prevTracks.map(track => 
        track.id === trackId 
          ? { ...track, duration: formatTime(duration) }
          : track
      )
    );
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.volume = isMuted ? 0 : volume;
  }, [volume, isMuted]);

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handlePlayPause = () => {
    if (!currentTrack) return;
    
    if (isPlaying) {
      audioRef.current?.pause();
    } else {
      audioRef.current?.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleTrackSelect = (track: AudioTrack) => {
    setCurrentTrack(track);
    setIsPlaying(false);
    setCurrentTime(0);
    
    // Load the actual audio file
    if (audioRef.current) {
      audioRef.current.src = track.file;
      audioRef.current.load();
      
      // Set duration when metadata is loaded
      audioRef.current.addEventListener('loadedmetadata', () => {
        const audioDuration = audioRef.current?.duration || 0;
        setDuration(audioDuration);
        updateTrackDuration(track.id, audioDuration);
      }, { once: true });
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const handleDownload = (track: AudioTrack) => {
    // In a real app, this would trigger a download
    console.log(`Downloading ${track.title}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Audio Demo Player
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300">
            Experience high-quality audio files from the public/audio folder
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Track List */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-6">
              <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-6">
                Available Tracks
              </h2>
              <div className="space-y-4">
                {tracks.map((track, index) => (
                  <motion.div
                    key={track.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`p-4 rounded-xl border-2 transition-all duration-300 cursor-pointer hover:shadow-lg ${
                      currentTrack?.id === track.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                    }`}
                    onClick={() => handleTrackSelect(track)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <h3 className="font-semibold text-slate-900 dark:text-white">
                          {track.title}
                        </h3>
                        <p className="text-slate-600 dark:text-slate-400 text-sm">
                          {track.artist}
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
                          <Clock className="w-4 h-4" />
                          <span>{track.duration}</span>
                        </div>
                        
                        <div className="px-2 py-1 rounded-md bg-slate-100 dark:bg-slate-700 text-xs font-medium text-slate-600 dark:text-slate-400">
                          {track.fileType.toUpperCase()}
                        </div>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownload(track);
                          }}
                          className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                        >
                          <Download className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                        </button>
                      </div>
                    </div>
                    
                    {/* Waveform Visualization */}
                    <div className="mt-3 flex items-center space-x-1">
                      {track.waveform.map((height, i) => (
                        <motion.div
                          key={i}
                          className="bg-slate-300 dark:bg-slate-600 rounded-full"
                          style={{
                            width: '3px',
                            height: `${height * 40}px`,
                          }}
                          animate={{
                            height: currentTrack?.id === track.id && isPlaying 
                              ? [`${height * 40}px`, `${height * 50}px`, `${height * 40}px`]
                              : `${height * 40}px`
                          }}
                          transition={{
                            duration: 0.5,
                            repeat: currentTrack?.id === track.id && isPlaying ? Infinity : 0,
                            delay: i * 0.1
                          }}
                        />
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Player Controls */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl p-6 sticky top-6">
              <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-6">
                Now Playing
              </h2>
              
              {currentTrack ? (
                <div className="space-y-6">
                  {/* Track Info */}
                  <div className="text-center">
                    <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                      {currentTrack.title}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400">
                      {currentTrack.artist}
                    </p>
                  </div>

                  {/* Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm text-slate-500 dark:text-slate-400">
                      <span>{formatTime(currentTime)}</span>
                      <span>{formatTime(duration)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max={duration || 100}
                      value={currentTime}
                      onChange={handleSeek}
                      className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                  </div>

                  {/* Control Buttons */}
                  <div className="flex items-center justify-center space-x-4">
                    <button className="p-3 rounded-full hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors">
                      <SkipBack className="w-6 h-6 text-slate-600 dark:text-slate-400" />
                    </button>
                    
                    <button
                      onClick={handlePlayPause}
                      className="p-4 rounded-full bg-blue-500 hover:bg-blue-600 text-white transition-colors shadow-lg"
                    >
                      {isPlaying ? (
                        <Pause className="w-8 h-8" />
                      ) : (
                        <Play className="w-8 h-8 ml-1" />
                      )}
                    </button>
                    
                    <button className="p-3 rounded-full hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors">
                      <SkipForward className="w-6 h-6 text-slate-600 dark:text-slate-400" />
                    </button>
                  </div>

                  {/* Volume Control */}
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={toggleMute}
                      className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                    >
                      {isMuted ? (
                        <VolumeX className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                      ) : (
                        <Volume2 className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                      )}
                    </button>
                    
                    <div className="flex-1 relative">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={isMuted ? 0 : volume}
                        onChange={handleVolumeChange}
                        className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-slate-200 dark:bg-slate-700 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <Play className="w-8 h-8 text-slate-400 ml-1" />
                  </div>
                  <p className="text-slate-500 dark:text-slate-400">
                    Select a track to start playing
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Hidden audio element for actual playback */}
        <audio ref={audioRef} style={{ display: 'none' }} />
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .slider::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
      `}</style>
    </div>
  );
}
