/**
 * K. Yamuna — 3D WebGL Background Engine
 * Three.js Interactive Cyber-Sphere, Floating Geometric Constellation & Particle Universe
 */

(function () {
  'use strict';

  // Check WebGL Support
  function isWebGLAvailable() {
    try {
      const canvas = document.createElement('canvas');
      return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch (e) {
      return false;
    }
  }

  const canvas = document.getElementById('webgl-canvas');
  if (!canvas || !isWebGLAvailable() || typeof THREE === 'undefined') {
    console.warn('WebGL or Three.js not supported. Falling back to ambient CSS background.');
    return;
  }

  // --- Scene, Camera, Renderer Setup ---
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x060813, 0.0012);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
  camera.position.z = 80;

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    alpha: true,
    antialias: true,
    powerPreference: 'high-performance'
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // --- Objects Group ---
  const worldGroup = new THREE.Group();
  scene.add(worldGroup);

  // --- 1. Central 3D Cyber-Icosahedron (Wireframe & Vertices) ---
  const icoGeometry = new THREE.IcosahedronGeometry(26, 1);
  
  // Wireframe Mesh
  const icoMaterial = new THREE.MeshBasicMaterial({
    color: 0x8b5cf6,
    wireframe: true,
    transparent: true,
    opacity: 0.18
  });
  const icoMesh = new THREE.Mesh(icoGeometry, icoMaterial);
  worldGroup.add(icoMesh);

  // Inner Core Sphere
  const coreGeometry = new THREE.SphereGeometry(14, 24, 24);
  const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0x06b6d4,
    wireframe: true,
    transparent: true,
    opacity: 0.12
  });
  const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
  worldGroup.add(coreMesh);

  // Glowing Points on Vertices
  const icoPointsMaterial = new THREE.PointsMaterial({
    color: 0x38bdf8,
    size: 2.2,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending
  });
  const icoPoints = new THREE.Points(icoGeometry, icoPointsMaterial);
  worldGroup.add(icoPoints);

  // Orbital Rings
  const createRing = (radius, tube, color, rotX, rotY) => {
    const ringGeo = new THREE.TorusGeometry(radius, tube, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.22,
      wireframe: true
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = rotX;
    ring.rotation.y = rotY;
    worldGroup.add(ring);
    return ring;
  };

  const ring1 = createRing(36, 0.25, 0x06b6d4, Math.PI / 3, Math.PI / 6);
  const ring2 = createRing(44, 0.2, 0x8b5cf6, -Math.PI / 4, Math.PI / 4);

  // --- 2. Interactive Star Particle Universe ---
  const particlesCount = 900;
  const positions = new Float32Array(particlesCount * 3);
  const colors = new Float32Array(particlesCount * 3);

  const cyanColor = new THREE.Color(0x06b6d4);
  const violetColor = new THREE.Color(0x8b5cf6);
  const emeraldColor = new THREE.Color(0x10b981);

  for (let i = 0; i < particlesCount; i++) {
    const i3 = i * 3;
    // Spread in spherical shell
    const radius = 60 + Math.random() * 160;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 2 - 1);

    positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    positions[i3 + 2] = radius * Math.cos(phi);

    // Color gradient variations
    const choice = Math.random();
    let pickedColor = cyanColor;
    if (choice > 0.65) pickedColor = violetColor;
    else if (choice > 0.45) pickedColor = emeraldColor;

    colors[i3] = pickedColor.r;
    colors[i3 + 1] = pickedColor.g;
    colors[i3 + 2] = pickedColor.b;
  }

  const particlesGeometry = new THREE.BufferGeometry();
  particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const particlesMaterial = new THREE.PointsMaterial({
    size: 2.0,
    vertexColors: true,
    transparent: true,
    opacity: 0.75,
    blending: THREE.AdditiveBlending
  });

  const particleSystem = new THREE.Points(particlesGeometry, particlesMaterial);
  scene.add(particleSystem);

  // --- Mouse Parallax & Scroll Reactivity ---
  let mouseX = 0;
  let mouseY = 0;
  let targetX = 0;
  let targetY = 0;
  let scrollY = 0;
  let targetScrollY = 0;

  const windowHalfX = window.innerWidth / 2;
  const windowHalfY = window.innerHeight / 2;

  window.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX - windowHalfX) * 0.0008;
    mouseY = (e.clientY - windowHalfY) * 0.0008;
  }, { passive: true });

  window.addEventListener('scroll', () => {
    scrollY = window.pageYOffset || document.documentElement.scrollTop;
  }, { passive: true });

  // Handle Resize
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  });

  // --- Animation Loop ---
  let clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);

    const elapsedTime = clock.getElapsedTime();

    // Lerp smooth mouse tracking
    targetX += (mouseX - targetX) * 0.05;
    targetY += (mouseY - targetY) * 0.05;

    // Smooth scroll interpolation
    targetScrollY += (scrollY - targetScrollY) * 0.06;

    // Rotate core geometry
    icoMesh.rotation.x = elapsedTime * 0.12;
    icoMesh.rotation.y = elapsedTime * 0.18;
    icoPoints.rotation.x = icoMesh.rotation.x;
    icoPoints.rotation.y = icoMesh.rotation.y;

    coreMesh.rotation.x = -elapsedTime * 0.15;
    coreMesh.rotation.y = -elapsedTime * 0.22;

    ring1.rotation.z = elapsedTime * 0.15;
    ring2.rotation.z = -elapsedTime * 0.2;

    // Slow gentle rotation for star field
    particleSystem.rotation.y = elapsedTime * 0.02;
    particleSystem.rotation.x = elapsedTime * 0.01;

    // Camera perspective shift based on mouse & scroll
    camera.position.x += (targetX * 35 - camera.position.x) * 0.05;
    camera.position.y += (-targetY * 35 - (targetScrollY * 0.04) - camera.position.y) * 0.05;
    camera.lookAt(scene.position);

    renderer.render(scene, camera);
  }

  animate();
})();
