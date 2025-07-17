document.addEventListener('DOMContentLoaded', function () {
            const slider = document.querySelector('.slider');
            const slides = document.querySelectorAll('.slide');
            const prevBtn = document.querySelector('.btn-prev');
            const nextBtn = document.querySelector('.btn-next');
            const dots = document.querySelectorAll('.dot');

            let currentSlide = 0;
            const slideCount = slides.length;
            let slideInterval;
            const intervalTime = 5000; // 5 seconds

            // Initialize slider
            function initSlider() {
                updateSlider();
                startSlideShow();

                // Event listeners
                prevBtn.addEventListener('click', prevSlide);
                nextBtn.addEventListener('click', nextSlide);

                // Dot navigation
                dots.forEach((dot, index) => {
                    dot.addEventListener('click', () => {
                        goToSlide(index);
                    });
                });

                // Pause on hover
                slider.addEventListener('mouseenter', pauseSlideShow);
                slider.addEventListener('mouseleave', startSlideShow);
            }

            // Update slider position
            function updateSlider() {
                slider.style.transform = `translateX(-${currentSlide * 100}%)`;

                // Update dots
                dots.forEach((dot, index) => {
                    dot.classList.toggle('active', index === currentSlide);
                });
            }

            // Next slide
            function nextSlide() {
                currentSlide = (currentSlide + 1) % slideCount;
                updateSlider();
                resetInterval();
            }

            // Previous slide
            function prevSlide() {
                currentSlide = (currentSlide - 1 + slideCount) % slideCount;
                updateSlider();
                resetInterval();
            }

            // Go to specific slide
            function goToSlide(index) {
                currentSlide = index;
                updateSlider();
                resetInterval();
            }

            // Start slideshow
            function startSlideShow() {
                slideInterval = setInterval(nextSlide, intervalTime);
            }

            // Pause slideshow
            function pauseSlideShow() {
                clearInterval(slideInterval);
            }

            // Reset interval timer
            function resetInterval() {
                pauseSlideShow();
                startSlideShow();
            }

            // Touch events for mobile
            let touchStartX = 0;
            let touchEndX = 0;

            slider.addEventListener('touchstart', (e) => {
                touchStartX = e.changedTouches[0].screenX;
                pauseSlideShow();
            }, { passive: true });

            slider.addEventListener('touchend', (e) => {
                touchEndX = e.changedTouches[0].screenX;
                handleSwipe();
                startSlideShow();
            }, { passive: true });

            function handleSwipe() {
                const threshold = 50;
                if (touchEndX < touchStartX - threshold) {
                    nextSlide();
                } else if (touchEndX > touchStartX + threshold) {
                    prevSlide();
                }
            }

            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowRight') {
                    nextSlide();
                } else if (e.key === 'ArrowLeft') {
                    prevSlide();
                }
            });

            // Initialize the slider
            initSlider();
        });