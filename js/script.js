// script.js — small interactions & light animations
document.addEventListener('DOMContentLoaded', function(){
  // fade-in on scroll
  const obs = new IntersectionObserver((entries) => {
    for (let e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
      }
    }
  }, {threshold: 0.12});
  document.querySelectorAll('.fade-up').forEach(el => obs.observe(el));

  // simple link placeholders open in new tab when href="#"
  document.querySelectorAll('a[href="#"]').forEach(a => {
    a.addEventListener('click', (ev) => {
      ev.preventDefault();
      alert('占位链接 — 你可以把它替换为你的 PDF / code repo 链接');
    });
  });

  // --- Results Carousel (Multiple carousels support) ---
  // 为每个轮播图创建独立的控制器
  const carousels = document.querySelectorAll(".results-carousel");
  
  carousels.forEach(carousel => {
    const track = carousel.querySelector(".carousel-track");
    const slides = Array.from(track.children);
    const nextButton = carousel.querySelector(".carousel-btn.next");
    const prevButton = carousel.querySelector(".carousel-btn.prev");
    
    let currentIndex = 0;

    // 初始化：确保只有第一张幻灯片显示
    function initCarousel() {
      slides.forEach((slide, index) => {
        slide.classList.toggle("active", index === 0);
      });
    }

    function updateSlide(index) {
      // 使用CSS类控制显示/隐藏
      slides.forEach((slide, i) => {
        slide.classList.toggle("active", i === index);
      });
    }

    // 只有当按钮存在时才添加事件监听器
    if (nextButton && prevButton) {
      nextButton.addEventListener("click", () => {
        currentIndex = (currentIndex + 1) % slides.length;
        updateSlide(currentIndex);
      });

      prevButton.addEventListener("click", () => {
        currentIndex = (currentIndex - 1 + slides.length) % slides.length;
        updateSlide(currentIndex);
      });
    }

    // 初始化轮播图
    initCarousel();
  });
});

// 复制BibTeX功能
function copyBibtex() {
  const bibtexText = document.querySelector('#bibtex .bibtex').textContent;
  const button = document.querySelector('.copy-btn');
  
  navigator.clipboard.writeText(bibtexText).then(() => {
    // 保存原始文本
    const originalText = button.textContent;
    // 更改按钮文本为"已复制"
    button.textContent = 'Copied!';
    
    // 2秒后恢复原始文本
    setTimeout(() => {
      button.textContent = originalText;
    }, 2000);
  }).catch(err => {
    console.error('复制失败: ', err);
    alert('复制失败，请手动复制');
  });
}