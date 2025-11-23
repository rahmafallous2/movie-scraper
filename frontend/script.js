
const DATA_URL = "https://raw.githubusercontent.com/rahmafallous2/movie-scraper/main/data/combined_movies.csv";

let allMovies = [];
let currentPage = 1;
const moviesPerPage = 20;
let filteredMovies = [];

async function loadMovieData() {
    try {
        showLoading();
        
        const response = await fetch(DATA_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const csvText = await response.text();
        const rows = parseCSV(csvText);
        
        allMovies = rows.slice(1).map(row => {
            const movie = {};
            rows[0].forEach((header, index) => {
                movie[header] = row[index] || '';
            });
            return movie;
        });
        
        updateStats();
        populateGenreFilter();
        filterMovies();
        
    } catch (error) {
        console.error('Error loading movie data:', error);
        document.getElementById('movies-body').innerHTML = 
            '<tr><td colspan="6" style="text-align: center; color: red;">Error loading data. Please check the console.</td></tr>';
    }
}

function parseCSV(csvText) {
    const rows = [];
    let currentRow = [];
    let inQuotes = false;
    let currentCell = '';
    
    for (let i = 0; i < csvText.length; i++) {
        const char = csvText[i];
        const nextChar = csvText[i + 1];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            currentRow.push(currentCell);
            currentCell = '';
        } else if (char === '\n' && !inQuotes) {
            currentRow.push(currentCell);
            rows.push(currentRow);
            currentRow = [];
            currentCell = '';
        } else if (char === '\r' && nextChar === '\n' && !inQuotes) {
            
        } else {
            currentCell += char;
        }
    }
    

    if (currentCell || currentRow.length > 0) {
        currentRow.push(currentCell);
        rows.push(currentRow);
    }
    
    return rows;
}

function updateStats() {
    const totalMovies = allMovies.length;
    const tmdbCount = allMovies.filter(m => m.source === 'TMDB').length;
    const imdbCount = allMovies.filter(m => m.source === 'IMDb').length;
    
   
    const dates = allMovies.map(m => m.combined_at || m.scraped_at).filter(Boolean);
    const latestDate = dates.length > 0 ? new Date(Math.max(...dates.map(d => new Date(d)))) : null;
    
    const statsHTML = `
        <div class="stat-card">
            <h3>Total Movies</h3>
            <p id="total-movies">${totalMovies.toLocaleString()}</p>
        </div>
        <div class="stat-card">
            <h3>TMDB Movies</h3>
            <p id="tmdb-count">${tmdbCount.toLocaleString()}</p>
        </div>
        <div class="stat-card">
            <h3>IMDb Movies</h3>
            <p id="imdb-count">${imdbCount.toLocaleString()}</p>
        </div>
        <div class="stat-card">
            <h3>Last Updated</h3>
            <p id="last-updated">${latestDate ? latestDate.toLocaleDateString() : 'Unknown'}</p>
        </div>
    `;
    
    document.getElementById('stats-container').innerHTML = statsHTML;
}

function populateGenreFilter() {
    const genreFilter = document.getElementById('genre-filter');
    const allGenres = new Set();
    
    allMovies.forEach(movie => {
        const genres = parseGenres(movie.genres);
        genres.forEach(genre => allGenres.add(genre));
    });
    
    const sortedGenres = Array.from(allGenres).sort();
    
    sortedGenres.forEach(genre => {
        const option = document.createElement('option');
        option.value = genre;
        option.textContent = genre;
        genreFilter.appendChild(option);
    });
}

function parseGenres(genresString) {
    if (!genresString) return [];
    
  
    try {
        const parsed = JSON.parse(genresString.replace(/'/g, '"'));
        return Array.isArray(parsed) ? parsed : [];
    } catch {
      
        return genresString.split(',').map(g => g.trim()).filter(g => g);
    }
}

function filterMovies() {
    const sourceFilter = document.getElementById('source-filter').value;
    const genreFilter = document.getElementById('genre-filter').value;
    const searchTerm = document.getElementById('search').value.toLowerCase();
    
    filteredMovies = allMovies.filter(movie => {
      
        if (sourceFilter !== 'all' && movie.source !== sourceFilter) {
            return false;
        }
        
      
        if (genreFilter !== 'all') {
            const genres = parseGenres(movie.genres);
            if (!genres.includes(genreFilter)) {
                return false;
            }
        }
        
 
        if (searchTerm) {
            const searchableText = `${movie.title} ${movie.overview}`.toLowerCase();
            if (!searchableText.includes(searchTerm)) {
                return false;
            }
        }
        
        return true;
    });
    
    currentPage = 1;
    renderMovies();
    updatePagination();
}

function renderMovies() {
    const moviesBody = document.getElementById('movies-body');
    const startIndex = (currentPage - 1) * moviesPerPage;
    const endIndex = startIndex + moviesPerPage;
    const pageMovies = filteredMovies.slice(startIndex, endIndex);
    
    if (pageMovies.length === 0) {
        moviesBody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No movies found matching your filters.</td></tr>';
        return;
    }
    
    moviesBody.innerHTML = pageMovies.map(movie => `
        <tr>
            <td><strong>${escapeHtml(movie.title)}</strong></td>
            <td>
                <span class="source-badge source-${movie.source?.toLowerCase() || 'unknown'}">
                    ${movie.source || 'Unknown'}
                </span>
            </td>
            <td>
                ${parseGenres(movie.genres).map(genre => 
                    `<span class="genre-tag">${escapeHtml(genre)}</span>`
                ).join('')}
            </td>
            <td>${movie.vote_average ? `${movie.vote_average}/10` : 'N/A'}</td>
            <td>${movie.release_date || 'Unknown'}</td>
            <td>${movie.vote_count ? parseInt(movie.vote_count).toLocaleString() : 'N/A'}</td>
        </tr>
    `).join('');
}

function updatePagination() {
    const totalPages = Math.ceil(filteredMovies.length / moviesPerPage);
    const pageInfo = document.getElementById('page-info');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    
    pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages || totalPages === 0;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading() {
    document.getElementById('movies-body').innerHTML = 
        '<tr><td colspan="6" style="text-align: center;">Loading movie data...</td></tr>';
}


document.getElementById('source-filter').addEventListener('change', filterMovies);
document.getElementById('genre-filter').addEventListener('change', filterMovies);
document.getElementById('search').addEventListener('input', filterMovies);

document.getElementById('prev-btn').addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        renderMovies();
        updatePagination();
    }
});

document.getElementById('next-btn').addEventListener('click', () => {
    const totalPages = Math.ceil(filteredMovies.length / moviesPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        renderMovies();
        updatePagination();
    }
});


document.addEventListener('DOMContentLoaded', loadMovieData);