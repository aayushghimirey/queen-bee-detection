"""Add predicted_frame_count to hive_predictions

Revision ID: 564a3efe1b68
Revises: 564a3efe1b67
Create Date: 2026-10-22 14:48:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '564a3efe1b68'
down_revision: Union[str, Sequence[str], None] = '564a3efe1b67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'hive_predictions',
        sa.Column('predicted_frame_count', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('hive_predictions', 'predicted_frame_count')
