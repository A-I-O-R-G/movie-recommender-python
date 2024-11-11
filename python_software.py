import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.DEFAULT_N_RECOMMENDATIONS = 5  # Valor padrão para recomendações

    def preprocess_data(self):
        """Prepara e transforma os dados de filmes e avaliações para o formato adequado."""
        try:
            movie_ratings = pd.merge(self.ratings, self.movies, on='movieId')
            user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)
            return user_movie_ratings
        except Exception as e:
            logging.error("Erro ao processar os dados: %s", e)
            raise

    def fit(self):
        """Treina o modelo de recomendação usando a similaridade de cosseno."""
        user_movie_ratings = self.preprocess_data()
        scaler = StandardScaler()
        normalized_ratings = scaler.fit_transform(user_movie_ratings)
        self.model = cosine_similarity(normalized_ratings)

    def content_based_recommend(self, movie_title, n_recommendations=None):
        """Gera recomendações baseadas em conteúdo."""
        if n_recommendations is None:
            n_recommendations = self.DEFAULT_N_RECOMMENDATIONS
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['title'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            idx = self.movies[self.movies['title'] == movie_title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            recommended_indices = [i[0] for i in sim_scores[1:min(n_recommendations + 1, len(sim_scores))]]
            recommended_movies = self.movies.iloc[recommended_indices]['title']
            return recommended_movies.tolist()
        except IndexError:
            logging.warning("Título do filme '%s' não encontrado.", movie_title)
            return []
        except Exception as e:
            logging.error("Erro ao gerar recomendações baseadas em conteúdo: %s", e)
            raise

    def recommend(self, user_id, n_recommendations=None):
        """Gera recomendações baseadas nas avaliações do usuário."""
        if self.model is None:
            logging.info("O modelo ainda não foi treinado. Treinando agora...")
            self.fit()

        if n_recommendations is None:
            n_recommendations = self.DEFAULT_N_RECOMMENDATIONS
        
        user_index = user_id - 1  # Ajusta o índice base
        sim_scores = list(enumerate(self.model[user_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommended_indices = [i[0] for i in sim_scores[1:min(n_recommendations + 1, len(sim_scores))]]
        recommended_movies = self.movies.iloc[recommended_indices]['title']
        return recommended_movies.tolist()

if __name__ == "__main__":
    # Exemplo de dados de filmes e avaliações (simulados)
    movies_data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Filme A', 'Filme B', 'Filme C', 'Filme D', 'Filme E']
    }
    
    ratings_data = {
        'userId': [1, 1, 2, 3, 3],
        'movieId': [1, 2, 2, 3, 4],
        'rating': [5, 4, 4, 3, 2]
    }

    movies = pd.DataFrame(movies_data)
    ratings = pd.DataFrame(ratings_data)

    recommender = MovieRecommender(movies, ratings)
    recommender.fit()

    user_id = 1
    recommendations = recommender.recommend(user_id)
    logging.info("Recomendações para o usuário %d: %s", user_id, recommendations)

    movie_title = 'Filme A'
    content_recommendations = recommender.content_based_recommend(movie_title)
    logging.info("Recomendações baseadas em conteúdo para '%s': %s", movie_title, content_recommendations)