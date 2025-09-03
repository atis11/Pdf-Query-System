import re
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

logger = logging.getLogger(__name__)

@dataclass
class QueryProfile:
    """Profile of a query with its characteristics and recommended parameters."""
    query_type: str
    complexity: str  # 'simple', 'moderate', 'complex'
    length: int
    keywords: List[str]
    entities: List[str]
    intent: str  # 'factual', 'analytical', 'comparative', 'summarization'
    
    # Recommended parameters
    chunk_size: int
    chunk_overlap: int
    top_k: int
    dense_weight: float
    temperature: float
    
    confidence: float  # How confident we are in these parameters

class QueryAnalyzer:
    """Analyzes user queries and recommends optimal parameters."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Query type patterns
        self.patterns = {
            'factual': [
                r'\b(what|who|when|where|how|which|why)\b',
                r'\b(define|definition|meaning|is|are|was|were)\b',
                r'\b(explain|describe|tell me about)\b'
            ],
            'analytical': [
                r'\b(analyze|analysis|compare|contrast|evaluate|assess)\b',
                r'\b(advantages|disadvantages|pros|cons|benefits|drawbacks)\b',
                r'\b(impact|effect|influence|relationship|correlation)\b'
            ],
            'comparative': [
                r'\b(compare|versus|vs|difference|similar|different)\b',
                r'\b(better|worse|best|worst|superior|inferior)\b',
                r'\b(same|alike|unlike|distinct|unique)\b'
            ],
            'summarization': [
                r'\b(summarize|summary|overview|gist|main points)\b',
                r'\b(key|important|essential|critical|major)\b',
                r'\b(brief|concise|short|quick)\b'
            ],
            'numerical': [
                r'\b(how many|how much|count|number|percentage|ratio)\b',
                r'\b(statistics|data|figures|numbers|amount|quantity)\b',
                r'\b(increase|decrease|growth|decline|trend)\b'
            ]
        }
        
        # Parameter recommendations based on query characteristics
        self.parameter_recommendations = {
            'factual': {
                'chunk_size': 512,
                'chunk_overlap': 64,
                'top_k': 3,
                'dense_weight': 0.7,
                'temperature': 0.3
            },
            'analytical': {
                'chunk_size': 768,
                'chunk_overlap': 128,
                'top_k': 7,
                'dense_weight': 0.6,
                'temperature': 0.5
            },
            'comparative': {
                'chunk_size': 1024,
                'chunk_overlap': 128,
                'top_k': 8,
                'dense_weight': 0.5,
                'temperature': 0.4
            },
            'summarization': {
                'chunk_size': 1024,
                'chunk_overlap': 128,
                'top_k': 10,
                'dense_weight': 0.4,
                'temperature': 0.6
            },
            'numerical': {
                'chunk_size': 512,
                'chunk_overlap': 64,
                'top_k': 5,
                'dense_weight': 0.8,
                'temperature': 0.2
            }
        }
    
    def analyze_query(self, query: str) -> QueryProfile:
        """Analyze a query and return a profile with recommended parameters."""
        logger.info(f"Analyzing query: {query[:50]}...")
        
        try:
            # Basic query characteristics
            query_lower = query.lower().strip()
            
            # Try NLTK tokenization first, fallback to simple split
            try:
                tokens = word_tokenize(query_lower)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed, using simple split: {e}")
                tokens = query_lower.split()
            
            length = len(tokens)
            
            # Remove stop words for keyword extraction
            keywords = [token for token in tokens if token.lower() not in self.stop_words and len(token) > 2]
            
            # Determine query type
            query_type, confidence = self._classify_query(query_lower)
            
            # Determine complexity
            complexity = self._assess_complexity(query_lower, length, keywords)
            
            # Determine intent
            intent = self._determine_intent(query_lower, query_type)
            
            # Extract entities (simple approach)
            entities = self._extract_entities(tokens)
            
            # Get recommended parameters
            params = self._get_recommended_parameters(query_type, complexity, length)
            
            profile = QueryProfile(
                query_type=query_type,
                complexity=complexity,
                length=length,
                keywords=keywords,
                entities=entities,
                intent=intent,
                chunk_size=params['chunk_size'],
                chunk_overlap=params['chunk_overlap'],
                top_k=params['top_k'],
                dense_weight=params['dense_weight'],
                temperature=params['temperature'],
                confidence=confidence
            )
            
            logger.info(f"Query analysis complete: {query_type} ({complexity}), confidence: {confidence:.2f}")
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return default profile if analysis fails
            return QueryProfile(
                query_type='factual',
                complexity='moderate',
                length=len(query.split()),
                keywords=[],
                entities=[],
                intent='factual',
                chunk_size=512,
                chunk_overlap=64,
                top_k=5,
                dense_weight=0.5,
                temperature=0.6,
                confidence=0.1
            )
    
    def _classify_query(self, query: str) -> Tuple[str, float]:
        """Classify the query type and return confidence score."""
        scores = {}
        
        for query_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            if score > 0:
                scores[query_type] = score
        
        if not scores:
            # Default to factual if no patterns match
            return 'factual', 0.5
        
        # Find the type with highest score
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        # Calculate confidence based on score strength
        total_possible = sum(len(patterns) for patterns in self.patterns.values())
        confidence = min(max_score / total_possible, 1.0)
        
        return best_type, confidence
    
    def _assess_complexity(self, query: str, length: int, keywords: List[str]) -> str:
        """Assess query complexity."""
        # Simple heuristics for complexity
        if length < 5 or len(keywords) < 2:
            return 'simple'
        elif length > 15 or len(keywords) > 8:
            return 'complex'
        else:
            return 'moderate'
    
    def _determine_intent(self, query: str, query_type: str) -> str:
        """Determine the intent of the query."""
        # Map query types to intents
        intent_mapping = {
            'factual': 'factual',
            'numerical': 'factual',
            'analytical': 'analytical',
            'comparative': 'comparative',
            'summarization': 'summarization'
        }
        return intent_mapping.get(query_type, 'factual')
    
    def _extract_entities(self, tokens: List[str]) -> List[str]:
        """Extract potential entities from the query."""
        # Simple entity extraction using POS tagging
        try:
            pos_tags = pos_tag(tokens)
            entities = []
            
            for token, pos in pos_tags:
                # Extract proper nouns, nouns, and numbers
                if pos in ['NNP', 'NNPS', 'NN', 'NNS', 'CD'] and len(token) > 2:
                    entities.append(token)
            
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _get_recommended_parameters(self, query_type: str, complexity: str, length: int) -> Dict:
        """Get recommended parameters based on query characteristics."""
        base_params = self.parameter_recommendations.get(query_type, self.parameter_recommendations['factual']).copy()
        
        # Adjust based on complexity
        if complexity == 'simple':
            base_params['chunk_size'] = max(256, base_params['chunk_size'] - 128)
            base_params['top_k'] = max(2, base_params['top_k'] - 2)
            base_params['temperature'] = min(0.8, base_params['temperature'] + 0.1)
        elif complexity == 'complex':
            base_params['chunk_size'] = min(1024, base_params['chunk_size'] + 128)
            base_params['top_k'] = min(10, base_params['top_k'] + 2)
            base_params['temperature'] = max(0.1, base_params['temperature'] - 0.1)
        
        # Adjust based on query length
        if length > 20:
            base_params['chunk_size'] = min(1024, base_params['chunk_size'] + 64)
            base_params['top_k'] = min(10, base_params['top_k'] + 1)
        
        return base_params
    
    def get_parameter_explanation(self, profile: QueryProfile) -> str:
        """Generate explanation for the recommended parameters."""
        explanations = {
            'factual': "Focused on precise, factual answers with smaller chunks for accuracy.",
            'analytical': "Balanced approach for analysis with moderate chunk size and overlap.",
            'comparative': "Larger chunks and more retrieval for comprehensive comparisons.",
            'summarization': "Maximum context with large chunks and high retrieval count.",
            'numerical': "Precise retrieval with high dense weight for accurate data."
        }
        
        base_explanation = explanations.get(profile.query_type, "Standard parameters for general queries.")
        
        complexity_explanation = {
            'simple': "Simple query - using smaller chunks and focused retrieval.",
            'moderate': "Moderate complexity - balanced parameters.",
            'complex': "Complex query - using larger chunks and broader retrieval."
        }
        
        return f"{base_explanation} {complexity_explanation.get(profile.complexity, '')}"
