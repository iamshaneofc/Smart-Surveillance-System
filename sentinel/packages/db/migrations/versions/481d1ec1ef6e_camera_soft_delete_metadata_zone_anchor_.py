"""0003 camera soft delete metadata zone anchor rule version site

Revision ID: 481d1ec1ef6e
Revises: dcfc0d2a87b2
Create Date: 2026-10-03 17:44:49.251962
"""
from alembic import op
import sqlalchemy as sa


revision = '481d1ec1ef6e'
down_revision = 'dcfc0d2a87b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('camera', schema=None) as batch_op:
        batch_op.add_column(sa.Column('metadata', sa.JSON(), nullable=False, server_default='{}'))
        batch_op.add_column(sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))

    with op.batch_alter_table('rule', schema=None) as batch_op:
        batch_op.add_column(sa.Column('site_id', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('version', sa.String(length=32), nullable=False, server_default='1'))
        batch_op.create_index(batch_op.f('ix_rule_site_id'), ['site_id'], unique=False)
        batch_op.create_foreign_key('fk_rule_site_id_site', 'site', ['site_id'], ['id'])

    with op.batch_alter_table('zone', schema=None) as batch_op:
        batch_op.add_column(sa.Column('anchor', sa.String(length=32), nullable=False, server_default='center'))


def downgrade() -> None:
    with op.batch_alter_table('zone', schema=None) as batch_op:
        batch_op.drop_column('anchor')

    with op.batch_alter_table('rule', schema=None) as batch_op:
        batch_op.drop_constraint('fk_rule_site_id_site', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_rule_site_id'))
        batch_op.drop_column('version')
        batch_op.drop_column('site_id')

    with op.batch_alter_table('camera', schema=None) as batch_op:
        batch_op.drop_column('deleted_at')
        batch_op.drop_column('metadata')
