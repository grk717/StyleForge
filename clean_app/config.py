from pydantic import BaseSettings, Field


class AppConfig(BaseSettings):

    # models
    DETECTOR_MODEL_PATH: str = Field(default='detector.pt', env='DETECTOR_MODEL_PATH')
    EMBEDDER_MODEL_PATH: str = Field(default='resnet50.pth', env='EMBEDDER_MODEL_PATH')
    EMBEDDING_DATABASE_PATH: str = Field(default='embeddings_resnet.npy', env='EMBEDDING_DATABASE_PATH')
    DATABASE_PATH: str = Field(default='df_cropped.csv', env='DATABASE_PATH')

    # app config
    PORT: int = Field(default=5001, env='PORT')
    APP_NAME: str = Field(default='cv-api', env='APP_NAME')
    DEBUG: bool = Field(default='True', env='DEBUG')
