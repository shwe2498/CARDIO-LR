# gnn/train_rgcn.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse
from torch.optim import Adam
import sys
import time

# Add parent directory to path to properly import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED_KG_DIR, DATA_PROCESSED_MODELS_DIR

# Simplified RGCN model that doesn't rely on PyTorch Geometric
class SimpleRGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim=128, hidden_dim=256):
        super(SimpleRGCN, self).__init__()
        # Force CPU usage due to CUDA capability mismatch
        self.device = torch.device('cpu')
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.num_relations = num_relations
        
        # Create relation-specific weight matrices
        self.weight1 = nn.Parameter(torch.Tensor(num_relations, embedding_dim, hidden_dim))
        self.weight2 = nn.Parameter(torch.Tensor(num_relations, hidden_dim, embedding_dim))
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, node_ids, edge_index, edge_type):
        # Get node embeddings
        x = self.embedding(node_ids)
        
        # First layer
        x1 = self._propagate(x, edge_index, edge_type, self.weight1)
        x1 = F.relu(self.layer_norm1(x1))
        x1 = self.dropout(x1)
        
        # Second layer
        x2 = self._propagate(x1, edge_index, edge_type, self.weight2)
        x2 = self.layer_norm2(x2)
        
        # Skip connection
        x_final = x2 + x
        
        return x_final
    
    def _propagate(self, x, edge_index, edge_type, weight):
        # Simple message passing implementation
        src, dst = edge_index[0], edge_index[1]
        
        # Initialize output tensor with appropriate dimensions
        if weight.shape[-1] != x.shape[-1]:
            out = torch.zeros(x.shape[0], weight.shape[-1], device=self.device)
        else:
            out = torch.zeros_like(x, device=self.device)
        
        # Group edges by relation type for more efficient computation
        for rel in range(self.num_relations):
            # Find edges with this relation type
            mask = (edge_type == rel)
            if not mask.any():
                continue
                
            # Get source and destination nodes for this relation
            src_rel = src[mask]
            dst_rel = dst[mask]
            
            # Get embeddings for source nodes
            src_embedding = x[src_rel]
            
            # Apply relation-specific transformation
            transformed = torch.matmul(src_embedding, weight[rel])
            
            # Aggregate messages for each destination node
            for i in range(len(dst_rel)):
                out[dst_rel[i]] += transformed[i]
        
        return out
    
    def get_embeddings(self, node_ids, edge_index, edge_type):
        return self.forward(node_ids, edge_index, edge_type)
    
    def train_model(self, train_data, optimizer, num_epochs=10, batch_size=128, save_path=None):
        """Train the RGCN model"""
        node_ids, edge_index, edge_type, pos_samples, neg_samples = train_data
        
        # Number of batches
        num_samples = len(pos_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            # Shuffle training data
            indices = torch.randperm(num_samples)
            pos_samples_shuffled = pos_samples[indices]
            neg_samples_shuffled = neg_samples[indices]
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx in progress_bar:
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                
                batch_pos = pos_samples_shuffled[start_idx:end_idx]
                batch_neg = neg_samples_shuffled[start_idx:end_idx]
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Get embeddings for all nodes
                node_embeddings = self.forward(node_ids, edge_index, edge_type)
                
                # Compute scores for positive and negative samples
                pos_scores = self._score_triples(node_embeddings, batch_pos)
                neg_scores = self._score_triples(node_embeddings, batch_neg)
                
                # Compute margin loss
                loss = F.margin_ranking_loss(
                    pos_scores, 
                    neg_scores, 
                    torch.ones(pos_scores.size(0), device=self.device),
                    margin=1.0
                )
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save model if loss improved
            if avg_loss < best_loss and save_path:
                best_loss = avg_loss
                self.save_model(save_path)
                print(f"Model saved to {save_path} (loss: {best_loss:.4f})")
    
    def _score_triples(self, node_embeddings, triples):
        """Compute similarity scores for triples"""
        heads = node_embeddings[triples[:, 0]]
        tails = node_embeddings[triples[:, 2]]
        
        # Simple dot product similarity
        scores = torch.sum(heads * tails, dim=1)
        return scores
    
    def save_model(self, path):
        """Save model state to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        """Load model state from disk"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

def load_kg(kg_path):
    """Load the knowledge graph from disk"""
    print(f"Loading knowledge graph from {kg_path}...")
    start_time = time.time()
    kg = nx.read_gpickle(kg_path)
    load_time = time.time() - start_time
    print(f"Knowledge graph loaded with {len(kg.nodes)} nodes and {len(kg.edges)} edges in {load_time:.2f} seconds")
    return kg

def prepare_training_data(kg, batch_size=5000, neg_samples_ratio=1, device=None):
    """
    Prepare training data for RGCN from the knowledge graph
    Returns full graph in PyTorch format and training samples
    """
    # Force CPU usage due to CUDA capability mismatch
    device = torch.device('cpu')
    
    # Create node mapping for the graph
    node_mapping = {node: i for i, node in enumerate(kg.nodes())}
    
    # Create relation mapping
    relation_types = set()
    for _, _, data in kg.edges(data=True):
        rel_type = data.get('relation', 'related_to')
        relation_types.add(rel_type)
    
    rel_to_idx = {rel: i for i, rel in enumerate(sorted(relation_types))}
    
    # Prepare edge index and type arrays
    src_nodes = []
    dst_nodes = []
    edge_types = []
    
    # Process all edges for the graph structure
    print("Converting graph to PyTorch format...")
    for src, dst, data in tqdm(kg.edges(data=True), total=len(kg.edges())):
        # Convert to indices
        src_idx = node_mapping[src]
        dst_idx = node_mapping[dst]
        
        # Get relation type
        rel_type = data.get('relation', 'related_to')
        rel_idx = rel_to_idx[rel_type]
        
        # Add to arrays
        src_nodes.append(src_idx)
        dst_nodes.append(dst_idx)
        edge_types.append(rel_idx)
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)
    edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
    
    # Node IDs tensor
    node_ids = torch.arange(len(node_mapping), device=device)
    
    # Create training samples
    print("Creating training samples...")
    
    # For large graphs, sample a subset of edges to use as positive samples
    num_edges = len(kg.edges())
    if num_edges > batch_size:
        edge_subset = list(kg.edges(data=True))
        selected_edges = np.random.choice(len(edge_subset), batch_size, replace=False)
        edges_for_training = [edge_subset[i] for i in selected_edges]
    else:
        edges_for_training = kg.edges(data=True)
    
    # Create positive samples from the selected edges
    positive_samples = []
    for src, dst, data in tqdm(edges_for_training, desc="Generating positive samples"):
        # Convert to indices
        src_idx = node_mapping[src]
        dst_idx = node_mapping[dst]
        
        # Get relation type
        rel_type = data.get('relation', 'related_to')
        rel_idx = rel_to_idx[rel_type]
        
        positive_samples.append([src_idx, rel_idx, dst_idx])
    
    # Create negative samples by corrupting heads or tails
    num_nodes = len(node_mapping)
    negative_samples = []
    
    print("Generating negative samples...")
    num_neg_samples = int(len(positive_samples) * neg_samples_ratio)
    
    for _ in tqdm(range(num_neg_samples), desc="Generating negative samples"):
        # Select a random positive sample
        pos_idx = np.random.randint(0, len(positive_samples))
        h, r, t = positive_samples[pos_idx]
        
        # Corrupt head or tail
        if np.random.random() < 0.5:
            # Corrupt head
            corrupt_h = h
            while corrupt_h == h:
                corrupt_h = np.random.randint(0, num_nodes)
            negative_samples.append([corrupt_h, r, t])
        else:
            # Corrupt tail
            corrupt_t = t
            while corrupt_t == t:
                corrupt_t = np.random.randint(0, num_nodes)
            negative_samples.append([h, r, corrupt_t])
    
    # Convert to PyTorch tensors
    positive_samples = torch.tensor(positive_samples, dtype=torch.long, device=device)
    negative_samples = torch.tensor(negative_samples, dtype=torch.long, device=device)
    
    return node_ids, edge_index, edge_type, positive_samples, negative_samples, len(rel_to_idx)

def train_rgcn(kg_path=None, output_path=None, embedding_dim=128, hidden_dim=256, 
               batch_size=128, num_epochs=10, learning_rate=0.001, neg_samples_ratio=1):
    """Train the RGCN model on the knowledge graph"""
    # Force CPU usage due to CUDA capability mismatch
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Default paths
    if kg_path is None:
        kg_path = os.path.join(DATA_PROCESSED_KG_DIR, 'integrated_cardio_graph.pkl')
    
    if output_path is None:
        output_path = os.path.join(DATA_PROCESSED_MODELS_DIR, 'rgcn_model.pt')
    
    # Load knowledge graph
    kg = load_kg(kg_path)
    
    # Prepare training data
    node_ids, edge_index, edge_type, pos_samples, neg_samples, num_relations = prepare_training_data(
        kg, batch_size=5000, neg_samples_ratio=neg_samples_ratio, device=device
    )
    
    # Create model
    num_nodes = len(kg.nodes())
    print(f"Creating RGCN model with {num_nodes} nodes and {num_relations} relation types")
    model = SimpleRGCN(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print(f"Training RGCN model for {num_epochs} epochs")
    train_data = (node_ids, edge_index, edge_type, pos_samples, neg_samples)
    model.train_model(train_data, optimizer, num_epochs=num_epochs, batch_size=batch_size, save_path=output_path)
    
    print(f"RGCN model trained and saved to {output_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RGCN model on knowledge graph")
    parser.add_argument("--kg_path", help="Path to knowledge graph pickle file")
    parser.add_argument("--output_path", help="Path to save trained model")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--neg_samples_ratio", type=float, default=1, help="Ratio of negative to positive samples")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(DATA_PROCESSED_MODELS_DIR, exist_ok=True)
    
    # Train the model
    train_rgcn(
        kg_path=args.kg_path,
        output_path=args.output_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        neg_samples_ratio=args.neg_samples_ratio
    )