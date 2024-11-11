import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.model = None

    def preprocess_data(self):
        # Combina os dados de filmes e avaliações
        movie_ratings = pd.merge(self.ratings, self.movies, on='movieId')
        user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)
        return user_movie_ratings

    def fit(self):
        user_movie_ratings = self.preprocess_data()
        # Normaliza os dados
        scaler = StandardScaler()
        normalized_ratings = scaler.fit_transform(user_movie_ratings)
        self.model = cosine_similarity(normalized_ratings)

    def recommend(self, user_id, n_recommendations=5):
        if self.model is None:
            print("O modelo ainda não foi treinado. Treinando agora...")
            self.fit()

        # Identifica o índice do usuário
        user_index = user_id - 1  # Ajusta o índice base
        sim_scores = list(enumerate(self.model[user_index]))
        
        # Ordena as recomendações
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Os indices dos filmes recomendados
        recommended_indices = [i[0] for i in sim_scores[1:n_recommendations + 1]]
        
        # Retorna os títulos dos filmes recomendados
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
    print(f"Recomendações para o usuário {user_id}: {recommendations}")