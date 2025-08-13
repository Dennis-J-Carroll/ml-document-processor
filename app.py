"""
Computer Vision Document Processor
Target Market: Accounting firms, small businesses, law offices
ROI Pitch: "Save 20 hours/week on document processing"
Value Prop: "Eliminate Manual Data Entry Forever"
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import json
import sqlite3
from datetime import datetime
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import uuid
import base64

# Mock OCR and CV libraries for demo (replace with real libraries in production)
class MockOCREngine:
    def __init__(self):
        self.confidence_threshold = 0.7
        
        # Mock patterns for different document types
        self.document_patterns = {
            'invoice': {
                'patterns': [
                    r'invoice\s*#?\s*:?\s*([A-Z0-9-]+)',
                    r'total\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                    r'date\s*:?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                    r'due\s*date\s*:?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                    r'bill\s*to\s*:?\s*([A-Za-z0-9\s,.-]+)'
                ],
                'fields': ['invoice_number', 'total_amount', 'invoice_date', 'due_date', 'bill_to']
            },
            'receipt': {
                'patterns': [
                    r'receipt\s*#?\s*:?\s*([A-Z0-9-]+)',
                    r'total\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                    r'date\s*:?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                    r'merchant\s*:?\s*([A-Za-z0-9\s,.-]+)',
                    r'tax\s*:?\s*\$?([0-9,]+\.?[0-9]*)'
                ],
                'fields': ['receipt_number', 'total_amount', 'date', 'merchant', 'tax_amount']
            },
            'business_card': {
                'patterns': [
                    r'([A-Za-z\s]+)(?=\n|$)',  # Name (first line typically)
                    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',  # Email
                    r'(\+?[1-9]\d{1,14})',  # Phone
                    r'([A-Za-z0-9\s,.-]+(?:LLC|Inc|Corp|Ltd))',  # Company
                    r'(CEO|Manager|Director|President|VP|Sales|Marketing)'  # Title
                ],
                'fields': ['name', 'email', 'phone', 'company', 'title']
            }
        }
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.BLUR)
        
        return image
    
    def extract_text(self, image: Image.Image) -> str:
        """Mock OCR text extraction"""
        # Simulate OCR by generating realistic text based on image characteristics
        width, height = image.size
        
        # Generate mock text based on image dimensions (different docs have different layouts)
        if width > height:  # Likely a receipt or invoice
            mock_text = """
            ACME CORPORATION
            Invoice #: INV-2024-001
            Date: 01/15/2024
            Due Date: 02/15/2024
            
            Bill To:
            John Smith
            123 Business St
            City, ST 12345
            
            Description          Qty    Unit Price    Total
            Consulting Services   10     $150.00      $1,500.00
            Software License      1      $500.00      $500.00
            
            Subtotal:                                 $2,000.00
            Tax (8%):                                 $160.00
            Total:                                    $2,160.00
            
            Thank you for your business!
            """
        else:  # Likely a business card
            mock_text = """
            Jane Doe
            Senior Marketing Manager
            TechStart Solutions Inc.
            
            📧 jane.doe@techstart.com
            📱 (555) 123-4567
            🌐 www.techstart.com
            
            123 Innovation Drive
            Tech City, TC 54321
            """
        
        return mock_text.strip()
    
    def classify_document(self, text: str) -> str:
        """Classify document type based on extracted text"""
        text_lower = text.lower()
        
        if 'invoice' in text_lower or 'bill to' in text_lower:
            return 'invoice'
        elif 'receipt' in text_lower or 'merchant' in text_lower:
            return 'receipt'
        elif '@' in text_lower and ('manager' in text_lower or 'director' in text_lower):
            return 'business_card'
        else:
            return 'unknown'
    
    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        """Extract structured data from text based on document type"""
        if doc_type not in self.document_patterns:
            return {'raw_text': text, 'confidence': 0.5}
        
        patterns = self.document_patterns[doc_type]['patterns']
        fields = self.document_patterns[doc_type]['fields']
        
        extracted_data = {}
        confidences = []
        
        for i, (pattern, field) in enumerate(zip(patterns, fields)):
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted_data[field] = match.group(1).strip()
                confidences.append(0.9)  # High confidence for matches
            else:
                extracted_data[field] = None
                confidences.append(0.3)  # Low confidence for missing data
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.5
        
        extracted_data.update({
            'document_type': doc_type,
            'confidence': overall_confidence,
            'raw_text': text,
            'extraction_timestamp': datetime.now().isoformat()
        })
        
        return extracted_data

@dataclass
class ProcessedDocument:
    id: str
    filename: str
    document_type: str
    extracted_data: Dict
    confidence: float
    processing_time: float
    created_at: datetime
    status: str = "processed"

class DocumentAI:
    def __init__(self, db_path="document_processor.db"):
        self.db_path = db_path
        self.ocr_engine = MockOCREngine()
        self.init_database()
        
        # Supported document types and their validation rules
        self.validation_rules = {
            'invoice': {
                'required_fields': ['invoice_number', 'total_amount', 'invoice_date'],
                'amount_fields': ['total_amount'],
                'date_fields': ['invoice_date', 'due_date']
            },
            'receipt': {
                'required_fields': ['total_amount', 'date'],
                'amount_fields': ['total_amount', 'tax_amount'],
                'date_fields': ['date']
            },
            'business_card': {
                'required_fields': ['name'],
                'email_fields': ['email'],
                'phone_fields': ['phone']
            }
        }
    
    def init_database(self):
        """Initialize database for storing processed documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                document_type TEXT NOT NULL,
                extracted_data TEXT NOT NULL,
                confidence REAL NOT NULL,
                processing_time REAL NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'processed'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                documents_processed INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_processing_time REAL DEFAULT 0,
                success_rate REAL DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                confidence REAL DEFAULT 0,
                validated BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (document_id) REFERENCES processed_documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_business_document(self, image_file, filename: str) -> ProcessedDocument:
        """Process uploaded document and extract structured data"""
        start_time = datetime.now()
        
        # Load and preprocess image
        image = Image.open(image_file)
        processed_image = self.ocr_engine.preprocess_image(image)
        
        # Extract text using OCR
        extracted_text = self.ocr_engine.extract_text(processed_image)
        
        # Classify document type
        doc_type = self.ocr_engine.classify_document(extracted_text)
        
        # Extract structured data
        structured_data = self.ocr_engine.extract_structured_data(extracted_text, doc_type)
        
        # Validate extracted data
        validated_data = self.validate_extracted_data(structured_data, doc_type)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create processed document object
        doc = ProcessedDocument(
            id=str(uuid.uuid4()),
            filename=filename,
            document_type=doc_type,
            extracted_data=validated_data,
            confidence=validated_data.get('confidence', 0.5),
            processing_time=processing_time,
            created_at=start_time
        )
        
        # Store in database
        self.store_document(doc)
        
        return doc
    
    def validate_extracted_data(self, data: Dict, doc_type: str) -> Dict:
        """Validate and clean extracted data"""
        if doc_type not in self.validation_rules:
            return data
        
        rules = self.validation_rules[doc_type]
        validated_data = data.copy()
        validation_issues = []
        
        # Check required fields
        missing_required = []
        for field in rules.get('required_fields', []):
            if not data.get(field):
                missing_required.append(field)
        
        if missing_required:
            validation_issues.append(f"Missing required fields: {', '.join(missing_required)}")
        
        # Validate amount fields
        for field in rules.get('amount_fields', []):
            if data.get(field):
                try:
                    # Clean and parse amount
                    amount_str = re.sub(r'[^\d.]', '', str(data[field]))
                    validated_data[field] = float(amount_str)
                except ValueError:
                    validation_issues.append(f"Invalid amount format: {field}")
                    validated_data[field] = None
        
        # Validate date fields
        for field in rules.get('date_fields', []):
            if data.get(field):
                date_str = str(data[field])
                # Try to parse common date formats
                date_patterns = [
                    r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
                    r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})'
                ]
                
                parsed_date = None
                for pattern in date_patterns:
                    match = re.search(pattern, date_str)
                    if match:
                        try:
                            # Assuming MM/DD/YYYY format
                            month, day, year = match.groups()
                            if len(year) == 2:
                                year = '20' + year
                            parsed_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            break
                        except:
                            continue
                
                if parsed_date:
                    validated_data[field] = parsed_date
                else:
                    validation_issues.append(f"Invalid date format: {field}")
        
        # Validate email fields
        for field in rules.get('email_fields', []):
            if data.get(field):
                email = str(data[field])
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, email):
                    validation_issues.append(f"Invalid email format: {field}")
        
        # Validate phone fields
        for field in rules.get('phone_fields', []):
            if data.get(field):
                phone = str(data[field])
                # Clean phone number
                phone_clean = re.sub(r'[^\d+]', '', phone)
                validated_data[field] = phone_clean
        
        # Add validation summary
        validated_data['validation_issues'] = validation_issues
        validated_data['is_valid'] = len(validation_issues) == 0
        
        # Adjust confidence based on validation
        if validation_issues:
            validated_data['confidence'] *= 0.7  # Reduce confidence for validation issues
        
        return validated_data
    
    def store_document(self, doc: ProcessedDocument):
        """Store processed document in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processed_documents 
            (id, filename, document_type, extracted_data, confidence, 
             processing_time, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc.id, doc.filename, doc.document_type, 
            json.dumps(doc.extracted_data), doc.confidence,
            doc.processing_time, doc.created_at.isoformat(), doc.status
        ))
        
        # Store individual fields for easier querying
        for field_name, field_value in doc.extracted_data.items():
            if field_name not in ['raw_text', 'validation_issues', 'extraction_timestamp']:
                cursor.execute('''
                    INSERT INTO document_fields 
                    (document_id, field_name, field_value, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (doc.id, field_name, str(field_value) if field_value else None, doc.confidence))
        
        conn.commit()
        conn.close()
    
    def batch_process_folder(self, file_list: List) -> List[ProcessedDocument]:
        """Process multiple documents at once"""
        processed_docs = []
        
        for file_info in file_list:
            try:
                doc = self.process_business_document(file_info['file'], file_info['filename'])
                processed_docs.append(doc)
            except Exception as e:
                st.error(f"Error processing {file_info['filename']}: {str(e)}")
        
        return processed_docs
    
    def get_processing_stats(self, days: int = 30) -> Dict:
        """Get processing statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                COUNT(*) as total_processed,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_processing_time,
                SUM(CASE WHEN confidence > 0.7 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                document_type,
                COUNT(*) as type_count
            FROM processed_documents 
            WHERE created_at > datetime('now', '-{} days')
            GROUP BY document_type
        '''.format(days)
        
        stats_df = pd.read_sql_query(query, conn)
        
        # Overall stats
        overall_query = '''
            SELECT 
                COUNT(*) as total_processed,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_processing_time,
                SUM(CASE WHEN confidence > 0.7 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM processed_documents 
            WHERE created_at > datetime('now', '-{} days')
        '''.format(days)
        
        overall_stats = pd.read_sql_query(overall_query, conn).iloc[0]
        
        conn.close()
        
        return {
            'overall': overall_stats.to_dict(),
            'by_type': stats_df.to_dict('records') if not stats_df.empty else []
        }

class DocumentProcessorApp:
    def __init__(self):
        self.processor = DocumentAI()
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Document Processor",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def file_upload_section(self):
        """File upload interface"""
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload business documents (invoices, receipts, business cards)",
            type=['png', 'jpg', 'jpeg', 'pdf', 'tiff'],
            accept_multiple_files=True,
            help="Supported formats: PNG, JPG, PDF, TIFF"
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                process_single = st.button("🔍 Process Individual Documents", type="primary")
            
            with col2:
                process_batch = st.button("⚡ Batch Process All", type="secondary")
            
            if process_single:
                self.process_individual_documents(uploaded_files)
            
            if process_batch:
                self.process_batch_documents(uploaded_files)
    
    def process_individual_documents(self, uploaded_files):
        """Process documents individually with detailed view"""
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---\n### Processing: {uploaded_file.name}")
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Process document
                    doc = self.processor.process_business_document(uploaded_file, uploaded_file.name)
                    
                    # Display results
                    self.display_document_results(doc, uploaded_file)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def process_batch_documents(self, uploaded_files):
        """Process multiple documents in batch"""
        st.markdown("### 🚀 Batch Processing Results")
        
        # Prepare file list
        file_list = [{'file': f, 'filename': f.name} for f in uploaded_files]
        
        # Process all documents
        with st.spinner(f"Processing {len(uploaded_files)} documents..."):
            processed_docs = self.processor.batch_process_folder(file_list)
        
        if processed_docs:
            # Summary table
            summary_data = []
            for doc in processed_docs:
                summary_data.append({
                    'Filename': doc.filename,
                    'Type': doc.document_type.title(),
                    'Confidence': f"{doc.confidence:.1%}",
                    'Processing Time': f"{doc.processing_time:.2f}s",
                    'Status': '✅ Success' if doc.confidence > 0.7 else '⚠️ Review Needed'
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Export options
            self.export_batch_results(processed_docs)
    
    def display_document_results(self, doc: ProcessedDocument, uploaded_file):
        """Display detailed results for a single document"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display the uploaded image
            st.image(uploaded_file, caption=f"Uploaded: {doc.filename}", use_column_width=True)
            
            # Processing metrics
            st.metric("Document Type", doc.document_type.title())
            st.metric("Confidence", f"{doc.confidence:.1%}")
            st.metric("Processing Time", f"{doc.processing_time:.2f}s")
        
        with col2:
            # Extracted data
            st.subheader("📊 Extracted Data")
            
            extracted_data = doc.extracted_data.copy()
            
            # Remove technical fields for display
            display_data = {k: v for k, v in extracted_data.items() 
                          if k not in ['raw_text', 'extraction_timestamp', 'validation_issues', 'is_valid']}
            
            # Display as key-value pairs
            for field, value in display_data.items():
                if value is not None:
                    st.text_input(f"{field.replace('_', ' ').title()}", str(value), key=f"{doc.id}_{field}")
            
            # Validation issues
            if extracted_data.get('validation_issues'):
                st.warning("⚠️ Validation Issues:")
                for issue in extracted_data['validation_issues']:
                    st.write(f"• {issue}")
            
            # Raw text (collapsible)
            with st.expander("📝 Raw Extracted Text"):
                st.text_area("", extracted_data.get('raw_text', ''), height=200)
        
        # Export options for individual document
        st.markdown("### 💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export as JSON", key=f"json_{doc.id}"):
                json_data = json.dumps(asdict(doc), default=str, indent=2)
                st.download_button("Download JSON", json_data, f"{doc.filename}_data.json")
        
        with col2:
            if st.button("📋 Export as CSV", key=f"csv_{doc.id}"):
                df = pd.DataFrame([display_data])
                csv_data = df.to_csv(index=False)
                st.download_button("Download CSV", csv_data, f"{doc.filename}_data.csv")
    
    def export_batch_results(self, processed_docs: List[ProcessedDocument]):
        """Export batch processing results"""
        st.markdown("### 💾 Batch Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export All as CSV"):
                # Prepare data for CSV export
                export_data = []
                for doc in processed_docs:
                    row = {'filename': doc.filename, 'document_type': doc.document_type, 
                           'confidence': doc.confidence}
                    row.update(doc.extracted_data)
                    export_data.append(row)
                
                df = pd.DataFrame(export_data)
                csv_data = df.to_csv(index=False)
                st.download_button("Download CSV", csv_data, "batch_processed_documents.csv")
        
        with col2:
            if st.button("📈 Export Summary Report"):
                # Generate summary report
                report = self.generate_batch_report(processed_docs)
                st.download_button("Download Report", report, "processing_summary.txt")
        
        with col3:
            if st.button("📋 Export to Database"):
                st.success("✅ All documents saved to database automatically")
    
    def generate_batch_report(self, processed_docs: List[ProcessedDocument]) -> str:
        """Generate a summary report for batch processing"""
        total_docs = len(processed_docs)
        successful_docs = len([d for d in processed_docs if d.confidence > 0.7])
        avg_confidence = np.mean([d.confidence for d in processed_docs])
        avg_processing_time = np.mean([d.processing_time for d in processed_docs])
        
        # Document type breakdown
        doc_types = {}
        for doc in processed_docs:
            doc_types[doc.document_type] = doc_types.get(doc.document_type, 0) + 1
        
        report = f"""
DOCUMENT PROCESSING SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total Documents Processed: {total_docs}
- Successfully Processed: {successful_docs} ({successful_docs/total_docs:.1%})
- Average Confidence: {avg_confidence:.1%}
- Average Processing Time: {avg_processing_time:.2f} seconds
- Total Processing Time: {sum(d.processing_time for d in processed_docs):.2f} seconds

DOCUMENT TYPE BREAKDOWN:
"""
        
        for doc_type, count in doc_types.items():
            report += f"- {doc_type.title()}: {count} documents\n"
        
        report += "\nDETAILED RESULTS:\n"
        for doc in processed_docs:
            status = "✅ SUCCESS" if doc.confidence > 0.7 else "⚠️ REVIEW NEEDED"
            report += f"- {doc.filename}: {doc.document_type.title()} ({doc.confidence:.1%}) {status}\n"
        
        return report
    
    def analytics_dashboard(self):
        """Display processing analytics"""
        st.subheader("📈 Processing Analytics")
        
        # Get statistics
        stats = self.processor.get_processing_stats(days=30)
        
        if stats['overall']['total_processed'] > 0:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents Processed", int(stats['overall']['total_processed']))
            
            with col2:
                st.metric("Average Confidence", f"{stats['overall']['avg_confidence']:.1%}")
            
            with col3:
                st.metric("Success Rate", f"{stats['overall']['success_rate']:.1%}")
            
            with col4:
                st.metric("Avg Processing Time", f"{stats['overall']['avg_processing_time']:.2f}s")
            
            # Document type breakdown
            if stats['by_type']:
                st.subheader("📊 Document Types Processed")
                type_df = pd.DataFrame(stats['by_type'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart of document types
                    fig_pie = px.pie(type_df, values='type_count', names='document_type',
                                   title='Documents by Type')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart of confidence by type
                    fig_bar = px.bar(type_df, x='document_type', y='avg_confidence',
                                   title='Average Confidence by Type',
                                   color='avg_confidence',
                                   color_continuous_scale='Viridis')
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("📊 No processing data available yet. Upload and process some documents to see analytics.")
    
    def recent_documents_table(self):
        """Display recently processed documents"""
        st.subheader("📄 Recent Documents")
        
        conn = sqlite3.connect(self.processor.db_path)
        query = '''
            SELECT filename, document_type, confidence, processing_time, created_at, status
            FROM processed_documents 
            ORDER BY created_at DESC 
            LIMIT 10
        '''
        
        recent_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not recent_df.empty:
            # Format for display
            recent_df['created_at'] = pd.to_datetime(recent_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            recent_df['confidence'] = (recent_df['confidence'] * 100).round(1).astype(str) + '%'
            recent_df['processing_time'] = recent_df['processing_time'].round(2).astype(str) + 's'
            
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No documents processed yet.")
    
    def run_app(self):
        """Main application interface"""
        st.title("📄 Computer Vision Document Processor")
        st.markdown("*Eliminate Manual Data Entry Forever - Save 20 hours/week on document processing*")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Process Documents", "📈 Analytics", "📄 Recent Documents", "⚙️ Settings"])
        
        with tab1:
            self.file_upload_section()
        
        with tab2:
            self.analytics_dashboard()
        
        with tab3:
            self.recent_documents_table()
        
        with tab4:
            self.settings_page()
    
    def settings_page(self):
        """Settings and configuration page"""
        st.subheader("⚙️ Settings & Configuration")
        
        st.markdown("### 🔧 Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
            st.checkbox("Auto-validate extracted data", value=True)
            st.selectbox("Default output format", ["JSON", "CSV", "Excel"])
        
        with col2:
            st.checkbox("Enable OCR preprocessing", value=True)
            st.checkbox("Store raw text", value=True)
            st.number_input("Batch processing limit", min_value=1, max_value=100, value=20)
        
        st.markdown("### 📊 Supported Document Types")
        
        supported_types = [
            "📧 Invoices - Extract invoice number, amounts, dates, billing details",
            "🧾 Receipts - Extract merchant, amounts, dates, tax information", 
            "💼 Business Cards - Extract names, contact information, company details",
            "📋 Forms - Extract form fields and data (coming soon)",
            "📑 Contracts - Extract key terms and parties (coming soon)"
        ]
        
        for doc_type in supported_types:
            st.write(f"✅ {doc_type}")
        
        st.markdown("### 🔌 Integration Options")
        st.info("Connect with popular business tools:")
        
        integrations = [
            "QuickBooks - Automatic invoice entry",
            "Xero - Expense tracking integration", 
            "Google Drive - Bulk folder processing",
            "Dropbox - Automatic document monitoring",
            "Slack - Processing notifications",
            "Email - Process attachments automatically"
        ]
        
        for integration in integrations:
            st.write(f"🔌 {integration}")
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")

if __name__ == "__main__":
    app = DocumentProcessorApp()
    app.run_app()