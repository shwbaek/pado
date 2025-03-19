document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on the index page
  if (window.location.pathname.endsWith('/index.html') || 
      window.location.pathname.endsWith('/') || 
      window.location.pathname === '') {
    // Hide the sidebar logo on index page
    const sidebarLogo = document.querySelector('.sidebar-brand-logo');
    if (sidebarLogo) {
      sidebarLogo.style.display = 'none';
    }
  }
}); 