'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import * as THREE from 'three';

// 3D Background Component
function ThreeBackground() {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animationRef = useRef<number | null>(null);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create floating particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 300;
    const posArray = new Float32Array(particlesCount * 3);
    const colorArray = new Float32Array(particlesCount * 3);

    for (let i = 0; i < particlesCount * 3; i += 3) {
      posArray[i] = (Math.random() - 0.5) * 25;
      posArray[i + 1] = (Math.random() - 0.5) * 25;
      posArray[i + 2] = (Math.random() - 0.5) * 10;
      
      // Create rainbow colors
      const hue = (i / (particlesCount * 3)) * 360;
      const color = new THREE.Color().setHSL(hue / 360, 0.8, 0.6);
      colorArray[i] = color.r;
      colorArray[i + 1] = color.g;
      colorArray[i + 2] = color.b;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    // Create floating orbs with different colors
    const orbGeometry = new THREE.SphereGeometry(0.15, 32, 32);
    const orbs: THREE.Mesh[] = [];
    const orbColors = [0x4f46e5, 0x8b5cf6, 0xec4899, 0xf59e0b, 0x10b981];
    
    for (let i = 0; i < 20; i++) {
      const orbMaterial = new THREE.MeshBasicMaterial({
        color: orbColors[i % orbColors.length],
        transparent: true,
        opacity: 0.4,
      });
      
      const orb = new THREE.Mesh(orbGeometry, orbMaterial);
      orb.position.set(
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 8
      );
      orbs.push(orb);
      scene.add(orb);
    }

    // Create wave effect
    const waveGeometry = new THREE.PlaneGeometry(20, 20, 50, 50);
    const waveMaterial = new THREE.MeshBasicMaterial({
      color: 0x4f46e5,
      transparent: true,
      opacity: 0.1,
      wireframe: true,
    });
    const waveMesh = new THREE.Mesh(waveGeometry, waveMaterial);
    waveMesh.rotation.x = -Math.PI / 2;
    waveMesh.position.z = -5;
    scene.add(waveMesh);

    // Mouse tracking
    const handleMouseMove = (event: MouseEvent) => {
      mouseRef.current.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouseRef.current.y = -(event.clientY / window.innerHeight) * 2 + 1;
    };

    window.addEventListener('mousemove', handleMouseMove);

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      const time = Date.now() * 0.001;

      // Rotate particles
      particlesMesh.rotation.x += 0.0005;
      particlesMesh.rotation.y += 0.001;
      particlesMesh.rotation.z += 0.0003;

      // Animate orbs with mouse interaction
      orbs.forEach((orb, index) => {
        orb.position.y += Math.sin(time + index) * 0.008;
        orb.position.x += Math.cos(time + index * 0.5) * 0.008;
        orb.position.z += Math.sin(time + index * 0.3) * 0.005;
        orb.rotation.x += 0.02;
        orb.rotation.y += 0.015;
        
        // Mouse interaction
        const distanceFromMouse = Math.sqrt(
          Math.pow(orb.position.x - mouseRef.current.x * 10, 2) +
          Math.pow(orb.position.y - mouseRef.current.y * 10, 2)
        );
        
        if (distanceFromMouse < 3) {
          orb.scale.setScalar(1.5);
          (orb.material as THREE.MeshBasicMaterial).opacity = 0.8;
        } else {
          orb.scale.setScalar(1);
          (orb.material as THREE.MeshBasicMaterial).opacity = 0.4;
        }
      });

      // Animate wave
      const positions = waveGeometry.attributes.position.array as Float32Array;
      for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 1];
        positions[i + 2] = Math.sin(x + time) * Math.cos(y + time) * 0.5;
      }
      waveGeometry.attributes.position.needsUpdate = true;
      waveMesh.rotation.z += 0.001;

      // Camera movement
      camera.position.x += (mouseRef.current.x * 2 - camera.position.x) * 0.01;
      camera.position.y += (mouseRef.current.y * 2 - camera.position.y) * 0.01;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      const currentMountRef = mountRef.current;
      if (currentMountRef && renderer.domElement) {
        currentMountRef.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return <div ref={mountRef} className="fixed inset-0 -z-10" />;
}

// Animated Input Component
function AnimatedInput({ 
  value, 
  onChange, 
  placeholder, 
  type = "text",
  className = "" 
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  type?: string;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className={`relative ${className}`}
    >
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg blur-xl"
        animate={{
          scale: [1, 1.05, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      {type === "textarea" ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="relative w-full h-64 p-6 text-lg border-2 border-white/20 rounded-lg focus:outline-none focus:ring-4 focus:ring-blue-500/50 text-white bg-black/40 backdrop-blur-xl placeholder-gray-400 resize-none"
        />
      ) : (
        <div className="relative">
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="relative w-full p-4 text-lg border-2 border-white/20 rounded-lg focus:outline-none focus:ring-4 focus:ring-blue-500/50 text-white
              bg-gradient-to-r from-blue-900/80 to-purple-900/80
              appearance-none
              "
            style={{
              // fallback for browsers that don't support tailwind's bg-gradient
              backgroundColor: "#312e81",
            }}
          >
            {placeholder === "Duration" && (
              <>
                <option value="1">1 minutes</option>
                <option value="5">5 minutes</option>
                <option value="10">10 minutes</option>
                <option value="15">15 minutes</option>
                <option value="20">20 minutes</option>
                <option value="30">30 minutes</option>
                <option value="45">45 minutes</option>
                <option value="60">60 minutes</option>
              </>
            )}
            {placeholder === "Tone" && (
              <>
                <option value="casual">Casual</option>
                <option value="funny">Funny</option>
                <option value="serious">Serious</option>
                <option value="professional">Professional</option>
                <option value="educational">Educational</option>
              </>
            )}
            {placeholder === "Framework" && (
              <>
                <option value="mooncast">MoonCast</option>
                <option value="podagent">PodAgent</option>
                <option value="dia">DIA</option>
                <option value="mooncast+dia">MoonCast + DIA</option>
                <option value="podagent+dia">PodAgent + DIA</option>
              </>
            )}
          </select>
          {/* Custom dropdown arrow */}
          <div className="pointer-events-none absolute right-4 top-1/2 transform -translate-y-1/2 text-white text-xl">
            ▼
          </div>
        </div>
      )}
    </motion.div>
  );
}

// Animated Button Component
function AnimatedButton({ onClick, children, isGenerating = false }: { onClick: () => void; children: React.ReactNode; isGenerating?: boolean }) {
  const [isHovered, setIsHovered] = useState(false);
  const [isPressed, setIsPressed] = useState(false);

  return (
    <motion.button
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onMouseDown={() => setIsPressed(true)}
      onMouseUp={() => setIsPressed(false)}
      className="relative w-full px-8 py-6 text-xl font-bold text-white rounded-xl overflow-hidden group"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.8 }}
    >
      {/* Animated background with smoother transitions */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600"
        animate={{
          background: [
            "linear-gradient(90deg, #2563eb 0%, #7c3aed 50%, #ec4899 100%)",
            "linear-gradient(90deg, #7c3aed 0%, #ec4899 50%, #2563eb 100%)",
            "linear-gradient(90deg, #ec4899 0%, #2563eb 50%, #7c3aed 100%)",
            "linear-gradient(90deg, #2563eb 0%, #7c3aed 50%, #ec4899 100%)",
          ],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      
      {/* Enhanced shimmer effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
        animate={{
          x: ["-100%", "100%"],
        }}
        transition={{
          duration: 2.5,
          repeat: Infinity,
          ease: "easeInOut",
          repeatDelay: 1,
        }}
      />

      {/* Improved glow effect on hover */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-blue-400/40 via-purple-400/40 to-pink-400/40 rounded-xl blur-xl"
        animate={{
          scale: isHovered ? 1.15 : 1,
          opacity: isHovered ? 0.9 : 0,
        }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      />

      {/* Additional subtle pulse effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 rounded-xl"
        animate={{
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Particle burst effect */}
      {isPressed && (
        <>
          {[...Array(12)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-white rounded-full"
              initial={{
                x: 0,
                y: 0,
                opacity: 1,
                scale: 0,
              }}
              animate={{
                x: Math.cos((i * Math.PI * 2) / 12) * 120,
                y: Math.sin((i * Math.PI * 2) / 12) * 120,
                opacity: 0,
                scale: 1,
              }}
              transition={{
                duration: 0.8,
                ease: "easeOut",
                delay: i * 0.05,
              }}
            />
          ))}
        </>
      )}
      
      <span className="relative z-10 flex items-center justify-center gap-2">
        {/* Enhanced spinner for generating state */}
        {isGenerating && (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
          />
        )}
        {children}
      </span>
    </motion.button>
  );
}

export function PodcastGenerator() {
  const [input, setInput] = useState('');
  const [duration, setDuration] = useState('5');
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);

  const startGeneration = async () => {
    try {
      setIsGenerating(true);
      setError('');
      setStatus('Starting generation...');
      setResult(null);

      const response = await fetch('/api/MoonCast/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic: input,
          duration: parseInt(duration),
          seed: Math.floor(Math.random() * 1000000), // Random seed
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start generation');
      }

      const data = await response.json();
      console.log('Start API response:', data); // Debug log
      
      // Handle different possible response formats
      const jobId = data.id || data.jobId || data.job_id || data.taskId || data.task_id;
      
      if (!jobId) {
        console.error('No job ID found in response:', data);
        throw new Error('No job ID received from server');
      }
      
      setJobId(jobId);
      setStatus('Generation started! Click "Check Status" to see progress.');
      setIsGenerating(false);
    } catch (err) {
      console.error('Error starting generation:', err);
      setError(err instanceof Error ? err.message : 'Failed to start generation');
      setIsGenerating(false);
      setStatus('');
    }
  };

  const checkStatus = async () => {
    if (!jobId) {
      setError('No job ID available');
      return;
    }

    try {
      setIsCheckingStatus(true);
      setError('');
      
      console.log('Checking status for job ID:', jobId); // Debug log
      const response = await fetch(`/api/MoonCast/status/${jobId}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Status API error:', response.status, errorText);
        throw new Error(`Status check failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Status API response:', data); // Debug log
      
      if (data.status === 'COMPLETED') {
        setStatus('Generation completed!');
        setResult(data);
      } else if (data.status === 'FAILED') {
        setError('Generation failed');
        setStatus('');
      } else {
        setStatus(`Status: ${data.status || 'Processing...'}`);
      }
    } catch (err) {
      console.error('Error checking status:', err);
      setError(err instanceof Error ? err.message : 'Failed to check status');
    } finally {
      setIsCheckingStatus(false);
    }
  };

  const handleGenerate = () => {
    if (!input.trim()) {
      setError('Please enter a podcast topic');
      return;
    }
    startGeneration();
  };

  const resetForm = () => {
    setInput('');
    setDuration('5');
    setIsGenerating(false);
    setJobId(null);
    setStatus('');
    setError('');
    setResult(null);
    setIsCheckingStatus(false);
  };

  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen p-4 overflow-hidden">
      {/* 3D Background */}
      <ThreeBackground />
      
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/50 via-black/30 to-black/50" />
      
      <motion.div 
        className="relative z-10 w-full max-w-4xl space-y-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        {/* Header */}
        <motion.div 
          className="text-center space-y-4"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.h1 
            className="text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500"
            animate={{
              backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "linear",
            }}
            style={{
              backgroundSize: "200% 200%",
            }}
          >
            Podcast Generator
          </motion.h1>
          <motion.p 
            className="text-2xl text-gray-300 font-light"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            Create your perfect podcast with AI
          </motion.p>
        </motion.div>

        {/* Error display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 text-red-300"
          >
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </motion.div>
        )}

        {/* Status display */}
        {status && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-blue-500/20 border border-blue-500/50 rounded-lg p-4 text-blue-300"
          >
            <p className="font-semibold">Status:</p>
            <p>{status}</p>
            {jobId && (
              <p className="text-sm text-blue-400 mt-2">Job ID: {jobId}</p>
            )}
          </motion.div>
        )}

        {/* Result display */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 text-green-300"
          >
            <p className="font-semibold">Generation Complete!</p>
            <pre className="mt-2 text-sm overflow-auto max-h-64">
              {JSON.stringify(result, null, 2)}
            </pre>
          </motion.div>
        )}

        {/* Main form */}
        <motion.div 
          className="space-y-6"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          {/* Textarea */}
          <AnimatedInput
            value={input}
            onChange={setInput}
            placeholder="Enter your podcast topic or description here..."
            type="textarea"
          />

          {/* Controls row 1 */}
          <div className="flex gap-4">
            <AnimatedInput
              value={duration}
              onChange={setDuration}
              placeholder="Duration"
              className="flex-1"
            />
          </div>

          {/* Action buttons */}
          <div className="flex gap-4">
            <AnimatedButton onClick={handleGenerate} isGenerating={isGenerating}>
              {isGenerating ? (
                <motion.span
                  key="generating"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2"
                >
                  Generating Podcast...
                </motion.span>
              ) : (
                <motion.span
                  key="generate"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  Generate Podcast
                </motion.span>
              )}
            </AnimatedButton>
            
            {jobId && !result && (
              <motion.button
                onClick={checkStatus}
                disabled={isCheckingStatus}
                className="px-8 py-6 text-xl font-bold text-white bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:cursor-not-allowed rounded-xl transition-colors"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: isCheckingStatus ? 1 : 1.02 }}
                whileTap={{ scale: isCheckingStatus ? 1 : 0.98 }}
              >
                {isCheckingStatus ? (
                  <motion.span
                    className="flex items-center gap-2"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    />
                    Checking...
                  </motion.span>
                ) : (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    Check Status
                  </motion.span>
                )}
              </motion.button>
            )}
            
            {(result || error) && (
              <motion.button
                onClick={resetForm}
                className="px-8 py-6 text-xl font-bold text-white bg-gray-600 hover:bg-gray-700 rounded-xl transition-colors"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Start New
              </motion.button>
            )}
          </div>
        </motion.div>

        {/* Floating elements */}
        <motion.div
          className="absolute top-20 left-20 w-4 h-4 bg-blue-500 rounded-full blur-sm"
          animate={{
            y: [0, -20, 0],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute top-40 right-32 w-6 h-6 bg-purple-500 rounded-full blur-sm"
          animate={{
            y: [0, 30, 0],
            opacity: [0.3, 0.8, 0.3],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1,
          }}
        />
        <motion.div
          className="absolute bottom-40 left-32 w-3 h-3 bg-pink-500 rounded-full blur-sm"
          animate={{
            y: [0, -15, 0],
            opacity: [0.4, 0.9, 0.4],
          }}
          transition={{
            duration: 2.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2,
          }}
        />
        
        {/* Additional floating elements */}
        <motion.div
          className="absolute top-60 left-1/4 w-2 h-2 bg-cyan-400 rounded-full blur-sm"
          animate={{
            y: [0, 25, 0],
            x: [0, 10, 0],
            opacity: [0.3, 0.7, 0.3],
          }}
          transition={{
            duration: 3.5,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.5,
          }}
        />
        <motion.div
          className="absolute bottom-60 right-1/4 w-5 h-5 bg-yellow-400 rounded-full blur-sm"
          animate={{
            y: [0, -30, 0],
            x: [0, -15, 0],
            opacity: [0.2, 0.6, 0.2],
          }}
          transition={{
            duration: 4.2,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1.5,
          }}
        />
        <motion.div
          className="absolute top-1/3 right-20 w-3 h-3 bg-green-400 rounded-full blur-sm"
          animate={{
            y: [0, 20, 0],
            opacity: [0.4, 0.8, 0.4],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: 2.8,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 0.8,
          }}
        />
        
        {/* Animated border lines */}
        <motion.div
          className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent"
          animate={{
            scaleX: [0, 1, 0],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute bottom-0 right-0 w-1 h-full bg-gradient-to-b from-transparent via-purple-500 to-transparent"
          animate={{
            scaleY: [0, 1, 0],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2,
          }}
        />
      </motion.div>
    </div>
  );
} 